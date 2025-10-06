/*
 * Copyright (c) 2025-2025 Maftei Albert-Alexandru
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Edge detection filter
 *
 * @see https://en.wikipedia.org/wiki/Canny_edge_detector
 */

#include "libavutil/avstring.h"
#include "libavutil/bprint.h"
#include "libavutil/eval.h"
#include "libavutil/file.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/parseutils.h"
#include "avfilter.h"
#include "filters.h"
#include "audio.h"
#include "video.h"
/*
 * timemapwarp filter: remap PTS according to timemap.txt
 * Example usage:
 *   ffmpeg -i in.mp4 -vf timemapwarp=timemap.txt out.mp4
 */

#include "libavfilter/avfilter.h"
#include "libavfilter/formats.h"
#include "libavutil/opt.h"
#include "libavutil/log.h"
#include "libavutil/mem.h"
#include "libavutil/time.h"

static const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_GBRP,   AV_PIX_FMT_GBRAP,
    AV_PIX_FMT_ARGB,   AV_PIX_FMT_RGBA,
    AV_PIX_FMT_ABGR,   AV_PIX_FMT_BGRA,
    AV_PIX_FMT_0RGB,   AV_PIX_FMT_RGB0,
    AV_PIX_FMT_0BGR,   AV_PIX_FMT_BGR0,
    AV_PIX_FMT_RGB24,  AV_PIX_FMT_BGR24,
    AV_PIX_FMT_GRAY8,
    AV_PIX_FMT_YUV410P,
    AV_PIX_FMT_YUV444P,  AV_PIX_FMT_YUVJ444P,
    AV_PIX_FMT_YUV420P,  AV_PIX_FMT_YUVJ420P,
    AV_PIX_FMT_YUVA444P, AV_PIX_FMT_YUVA420P,
    AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_YUVA420P10LE,
    AV_PIX_FMT_YUV444P10LE, AV_PIX_FMT_YUVA444P10LE,
    AV_PIX_FMT_YUV420P12LE,
    AV_PIX_FMT_YUV444P12LE,
    AV_PIX_FMT_YUV444P16LE, AV_PIX_FMT_YUVA444P16LE,
    AV_PIX_FMT_YUV420P16LE, AV_PIX_FMT_YUVA420P16LE,
    AV_PIX_FMT_YUV444P9LE, AV_PIX_FMT_YUVA444P9LE,
    AV_PIX_FMT_YUV420P9LE, AV_PIX_FMT_YUVA420P9LE,
    AV_PIX_FMT_NONE
};

typedef struct {
    const AVClass *class;

    char *timemap_filename;   ///< Path to timemap.txt
    int64_t *pts_map;          ///< Input PTS values
    size_t pts_map_size;            ///< Loaded points
} WarpContext;

static av_cold int init_timemap(WarpContext *s, void *log_ctx)
{
    FILE *f;
    char line[256];
    int line_num = 0, ret = 0;

    if (!s->timemap_filename) {
        av_log(log_ctx, AV_LOG_ERROR, "No timemap file specified\n");
        return AVERROR(EINVAL);
    }

    f = fopen(s->timemap_filename, "r");
    if (!f) {
        av_log(log_ctx, AV_LOG_ERROR, "Could not open timemap file '%s'\n",
               s->timemap_filename);
        return AVERROR(errno);
    }

    while (fgets(line, sizeof(line), f)) {
        line_num++;
        ret = parse_timemap_line(s, line, line_num, log_ctx);
        if (ret < 0) {
            fclose(f);
            return ret;
        }
    }
    fclose(f);

    if (s->nb_points < 2) {
        av_log(log_ctx, AV_LOG_ERROR,
               "Timemap file '%s' must contain at least 2 points\n",
               s->timemap_filename);
        return AVERROR(EINVAL);
    }

    av_log(log_ctx, AV_LOG_INFO, "Loaded %d timemap points from %s\n",
           s->nb_points, s->timemap_filename);
    return 0;
}

/* --- core filter callbacks --- */

static av_cold int init(AVFilterContext *ctx)
{
    WarpContext *s = ctx->priv;
    int ret;
    if (!s->timemap_filename) {
        av_log(ctx, AV_LOG_ERROR,
               "Timemap must be specified\n");
        return AVERROR(EINVAL);
    }

    if (s->timemap_filename) {
        ret = av_file_map(s->timemap_filename,
                          &s->pts_map, &s->pts_map_size, 0, ctx);
        if (ret < 0)
            return ret;
        s->pts_map_size/=8;
    }
    return 0;
}

static inline int64_t warp_pts(int64_t in_pts, WarpContext *s)
{
    // Handle empty timemap
    if (s->pts_map_size == 0)
        return in_pts;

    // Handle out-of-bounds
    if (in_pts <= s->pts_map[0])
        return s->pts_map[1];
    if (in_pts >= s->pts_map[s->pts_map_size -2 ])
        return s->out_pts[s->pts_map_size - 1];

    // Binary search to find the interval [in_pts[i], in_pts[i+1]]
    int left = 0;
    int right = s->nb_points - 2; // we access i and i+1
    int mid;

    while (left <= right) {
        mid = (left + right) / 2;
        if (in_pts < s->pts_map[mid*2])
            right = mid - 1;
        else if (in_pts > s->pts_map[mid*2 +2])
            left = mid + 1;
        else
            break; // found the interval
    }

    int64_t in0 = s->pts_map[mid * 2];
    int64_t in1 = s->in_pts[mid * 2 + 2];
    int64_t out0 = s->out_pts[mid*2 + 1];
    int64_t out1 = s->out_pts[mid*2 + 3];

    // Linear interpolation
    double t = (double)(in_pts - in0) / (double)(in1 - in0);
    return out0 + (int64_t)((out1 - out0) * t);
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx = inlink->dst;
    WarpContext *s       = ctx->priv;

    // Convert frame PTS to AV_TIME_BASE units
    int64_t in_pts = av_rescale_q(frame->pts, inlink->time_base, (AVRational){1, AV_TIME_BASE});

    // Warp PTS using binary search and interpolation
    int64_t out_pts = warp_pts(in_pts, s);

    // Convert back to stream time base
    frame->pts = av_rescale_q(out_pts, (AVRational){1, AV_TIME_BASE}, inlink->time_base);

    return ff_filter_frame(ctx->outputs[0], frame);
}
\

static const AVOption warp_options[] = {
    { "timemap", "Path to timemap", offsetof(WarpContext, timemap_filename), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM },
    { NULL }
};

AVFILTER_DEFINE_CLASS(warp);


static const AVFilterPad warp_inputs[] = {
    { .name = "default", .type = AVMEDIA_TYPE_VIDEO, .filter_frame = filter_frame },
};

static const AVFilterPad warp_outputs[] = {
    { .name = "default", .type = AVMEDIA_TYPE_VIDEO },
};

const FFFilter ff_vf_warp = {
    .p.name          = "warp",
    .p.description   = NULL_IF_CONFIG_SMALL("Warp video PTS according to timemap file"),
    .p.priv_class    = &warp_class,
    .p.flags       = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
    .priv_size     = sizeof(WarpContext),
    .init          = init,
    FILTER_INPUTS(warp_inputs),
    FILTER_OUTPUTS(warp_outputs),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
};
