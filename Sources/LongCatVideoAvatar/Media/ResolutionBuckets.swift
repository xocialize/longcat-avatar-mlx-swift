//
//  ResolutionBuckets.swift
//
//  Port target: LongCat-Video `longcat_video/utils/bukcet_config.py`
//  (`ASPECT_RATIO_627`, `ASPECT_RATIO_960`) and the pipeline's
//  `get_condition_shape`.
//
//  The output size is not free: upstream picks, from a fixed table, the bucket
//  whose height/width ratio is closest to the reference image's, and the image
//  is then resized to cover it and centre-cropped.
//

import Foundation

public enum AvatarResolution: String, Sendable, CaseIterable {
    /// About 627² pixels of area; 480×832 for a 16:9 image. The model's native tier.
    case p480 = "480p"
    /// About 960² pixels of area; 736×1248 for a 16:9 image.
    case p720 = "720p"

    /// `(height / width ratio, height, width)`, ascending by ratio.
    var buckets: [(ratio: Double, height: Int, width: Int)] {
        switch self {
        case .p480:
            return [
                (0.26, 320, 1216), (0.31, 352, 1120), (0.38, 384, 1024), (0.43, 416, 960), (0.52, 448, 864),
                (0.58, 480, 832), (0.67, 512, 768), (0.74, 544, 736), (0.86, 576, 672), (0.95, 608, 640),
                (1.05, 640, 608), (1.17, 672, 576), (1.29, 704, 544), (1.35, 736, 544), (1.50, 768, 512),
                (1.67, 800, 480), (1.73, 832, 480), (2.00, 896, 448), (2.31, 960, 416), (2.58, 992, 384),
                (2.75, 1056, 384), (3.09, 1088, 352), (3.70, 1184, 320), (3.80, 1216, 320), (3.90, 1248, 320),
                (4.00, 1280, 320),
            ]
        case .p720:
            return [
                (0.25, 480, 1920), (0.29, 512, 1792), (0.32, 544, 1696), (0.36, 576, 1600), (0.40, 608, 1504),
                (0.49, 672, 1376), (0.54, 704, 1312), (0.59, 736, 1248), (0.69, 800, 1152), (0.74, 832, 1120),
                (0.82, 864, 1056), (0.88, 896, 1024), (0.94, 928, 992), (1.00, 960, 960), (1.07, 992, 928),
                (1.14, 1024, 896), (1.22, 1056, 864), (1.31, 1088, 832), (1.35, 1120, 832), (1.44, 1152, 800),
                (1.70, 1248, 736), (2.00, 1344, 672), (2.05, 1376, 672), (2.47, 1504, 608), (2.53, 1536, 608),
                (2.83, 1632, 576), (3.06, 1664, 544), (3.12, 1696, 544), (3.62, 1856, 512), (3.93, 1888, 480),
                (4.00, 1920, 480),
            ]
        }
    }

    /// The bucket closest to an image of `imageWidth × imageHeight` (PT
    /// `get_condition_shape`: the first bucket at the minimum ratio distance).
    public func size(imageWidth: Int, imageHeight: Int) -> (height: Int, width: Int) {
        let ratio = Double(imageHeight) / Double(imageWidth)
        var best = buckets[0]
        for bucket in buckets where abs(bucket.ratio - ratio) < abs(best.ratio - ratio) {
            best = bucket
        }
        return (best.height, best.width)
    }

    /// Whether `width × height` is one of this tier's buckets.
    public func contains(width: Int, height: Int) -> Bool {
        buckets.contains { $0.height == height && $0.width == width }
    }
}
