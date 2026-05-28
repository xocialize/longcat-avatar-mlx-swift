//
//  NumpyArraySmoke.swift
//
//  Verifies the .npy reader for the two dtypes the test suite actually
//  uses (fp32 + int32). Round-trips via Python-equivalent byte layouts.
//

import Foundation
import XCTest
import MLX
@testable import LongCatVideoAvatar

final class NumpyArraySmoke: XCTestCase {
    var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("NumpyArraySmoke-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDir)
    }

    /// Hand-roll a minimal .npy file with a Python-numpy-equivalent header.
    private func writeNpy(_ url: URL, descr: String, shape: [Int], payload: Data) throws {
        var header = "{'descr': '\(descr)', 'fortran_order': False, 'shape': "
        if shape.count == 1 {
            header += "(\(shape[0]),)"
        } else {
            header += "(" + shape.map(String.init).joined(separator: ", ") + ")"
        }
        header += ", }"
        // Pad header to 64-byte alignment with spaces + trailing \n.
        let prefixSize = 10  // magic(6) + version(2) + headerLen(2 for v1)
        var headerBytes = header.data(using: .utf8)!
        let target = ((prefixSize + headerBytes.count + 1 + 63) / 64) * 64
        let pad = target - prefixSize - headerBytes.count - 1
        headerBytes.append(Data(repeating: 0x20, count: pad))
        headerBytes.append(0x0A)  // newline

        var out = Data()
        out.append(contentsOf: [0x93])  // magic
        out.append(contentsOf: "NUMPY".utf8)
        out.append(contentsOf: [0x01, 0x00])  // version 1.0
        // header length (LE u16)
        let hl = UInt16(headerBytes.count)
        out.append(UInt8(hl & 0xff))
        out.append(UInt8((hl >> 8) & 0xff))
        out.append(headerBytes)
        out.append(payload)

        try out.write(to: url)
    }

    func testLoadsFp32ArrayCorrectly() throws {
        let floats: [Float] = [1.5, -2.25, 3.75, 4.0, -0.125, 6.0]
        var payload = Data()
        for f in floats {
            var bits = f.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { payload.append(contentsOf: $0) }
        }
        let url = tempDir.appendingPathComponent("test_f4.npy")
        try writeNpy(url, descr: "<f4", shape: [2, 3], payload: payload)

        let arr = try loadNumpy(url: url)
        XCTAssertEqual(arr.shape, [2, 3])
        XCTAssertEqual(arr.dtype, .float32)
        // Spot-check a couple of values.
        XCTAssertEqual(arr[0, 0].item(Float.self), 1.5, accuracy: 1e-6)
        XCTAssertEqual(arr[1, 2].item(Float.self), 6.0, accuracy: 1e-6)
    }

    func testLoadsInt32ArrayCorrectly() throws {
        let ints: [Int32] = [7, 11, 42, -1, 0, 256]
        var payload = Data()
        for i in ints {
            var v = i.littleEndian
            withUnsafeBytes(of: &v) { payload.append(contentsOf: $0) }
        }
        let url = tempDir.appendingPathComponent("test_i4.npy")
        try writeNpy(url, descr: "<i4", shape: [6], payload: payload)

        let arr = try loadNumpy(url: url)
        XCTAssertEqual(arr.shape, [6])
        XCTAssertEqual(arr.dtype, .int32)
        XCTAssertEqual(arr[3].item(Int32.self), -1)
        XCTAssertEqual(arr[5].item(Int32.self), 256)
    }

    func testRejectsUnsupportedDtype() throws {
        // <f8 (fp64) — not supported.
        let url = tempDir.appendingPathComponent("test_f8.npy")
        try writeNpy(
            url, descr: "<f8", shape: [1],
            payload: Data(repeating: 0, count: 8)
        )
        XCTAssertThrowsError(try loadNumpy(url: url)) { err in
            guard case NumpyError.unsupportedDescr(let s) = err else {
                XCTFail("Wrong error type: \(err)")
                return
            }
            XCTAssertEqual(s, "<f8")
        }
    }

    func testRejectsBadMagic() throws {
        let url = tempDir.appendingPathComponent("not_npy.txt")
        try Data("hello world".utf8).write(to: url)
        XCTAssertThrowsError(try loadNumpy(url: url)) { err in
            guard case NumpyError.badMagic = err else {
                XCTFail("Wrong error type: \(err)")
                return
            }
        }
    }
}
