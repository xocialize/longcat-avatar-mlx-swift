//
//  ImageInput.swift
//
//  Reference image → the pipeline's `[1, 3, 1, H, W]` tensor in `[-1, 1]`.
//  Mirrors PT `video_processor.preprocess(..., resize_mode="crop")`: resize so
//  the image covers the target, then centre-crop.
//

import CoreGraphics
import Foundation
import ImageIO
import MLX

public enum ImageInputError: Error, CustomStringConvertible {
    case undecodable
    case contextFailed(width: Int, height: Int)

    public var description: String {
        switch self {
        case .undecodable: return "the reference image could not be decoded"
        case .contextFailed(let width, let height): return "could not make a \(width)×\(height) bitmap"
        }
    }
}

public enum ImageInput {

    /// Decode PNG / JPEG / HEIC bytes, upright (EXIF orientation applied).
    public static func decode(_ data: Data) throws -> CGImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { throw ImageInputError.undecodable }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: 8192,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
            ?? CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw ImageInputError.undecodable }
        return image
    }

    /// `image` resized to cover `width × height`, centre-cropped, as RGB bytes
    /// (row-major, top row first, 3 bytes per pixel).
    public static func rgbBytes(_ image: CGImage, width: Int, height: Int) throws -> [UInt8] {
        guard let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { throw ImageInputError.contextFailed(width: width, height: height) }
        context.interpolationQuality = .high
        let scale = max(Double(width) / Double(image.width), Double(height) / Double(image.height))
        let drawWidth = Double(image.width) * scale, drawHeight = Double(image.height) * scale
        context.draw(image, in: CGRect(
            x: (Double(width) - drawWidth) / 2, y: (Double(height) - drawHeight) / 2,
            width: drawWidth, height: drawHeight))
        guard let base = context.data else { throw ImageInputError.contextFailed(width: width, height: height) }
        let rgba = base.bindMemory(to: UInt8.self, capacity: width * height * 4)
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for pixel in 0..<(width * height) {
            rgb[pixel * 3] = rgba[pixel * 4]
            rgb[pixel * 3 + 1] = rgba[pixel * 4 + 1]
            rgb[pixel * 3 + 2] = rgba[pixel * 4 + 2]
        }
        return rgb
    }

    /// The pipeline's reference-frame tensor: `[1, 3, 1, height, width]` in `[-1, 1]`.
    public static func tensor(_ image: CGImage, width: Int, height: Int) throws -> MLXArray {
        let rgb = try rgbBytes(image, width: width, height: height)
        let hwc = MLXArray(rgb, [height, width, 3]).asType(.float32) / 127.5 - 1.0
        return hwc.transposed(2, 0, 1).reshaped(1, 3, 1, height, width)
    }
}
