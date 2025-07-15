import zlib

def string_to_bits(s):
    result = []
    for c in s:
        bits = bin(c)[2:].rjust(8, '0')
        result.append(bits)
    return ''.join(result)

def bits_to_string(bit_string):
    byte_data = bytearray()
    for i in range(0, len(bit_string), 8):
        byte = bit_string[i:i+8]
        byte_data.append(int(byte, 2))
    return bytes(byte_data)

def compress_string_to_bits(input_string):
    # Step 1: Convert string to bytes
    input_bytes = input_string.encode('utf-8')
    # print(input_bytes,len(input_bytes))
    
    # Step 2: Compress the bytes using zlib
    compressed_bytes = zlib.compress(input_bytes)
    # print(compressed_bytes,len(compressed_bytes))
    
    # Step 3: Convert compressed bytes to 01 bit string
    bit_string = string_to_bits(compressed_bytes)
    # print(bit_string,len(bit_string))
    
    return bit_string

def decompress_bits_to_string(bit_string):
    # Step 1: Convert 01 bit string to bytes
    compressed_bytes = bits_to_string(bit_string)
    
    # Step 2: Decompress the bytes using zlib
    decompressed_bytes = zlib.decompress(compressed_bytes)
    
    # Step 3: Convert decompressed bytes to string
    decompressed_string = decompressed_bytes.decode('utf-8')
    
    return decompressed_string

# Example usage

# message = '3D Gaussian Splatting (3DGS) has already become the emerging research focus in the fields of 3D scene reconstruction and novel view synthesis. Given that training a 3DGS requires a significant amount of time and computational cost, it is crucial to protect the copyright, integrity, and privacy of such 3D assets. Steganography, as a crucial technique for encrypted transmission and copyright protection, has been extensively studied. However, it still lacks profound exploration targeted at 3DGS. Unlike its predecessor NeRF, 3DGS possesses two distinct features: 1) explicit 3D representation; and 2) real-time rendering speeds. These characteristics result in the 3DGS point cloud files being public and transparent, with each Gaussian point having a clear physical significance. Therefore, ensuring the security and fidelity of the original 3D scene while embedding information into the 3DGS point cloud files is an extremely challenging task. To solve the above-mentioned issue, we first propose a steganography framework for 3DGS, dubbed GS-Hider, which can embed 3D scenes and images into original GS point clouds in an invisible manner and accurately extract the hidden messages.' 

# input_string = message

# # Compress the string to 01 bit string
# bit_string, compressed_bytes = compress_string_to_bits(input_string)

# # Decompress the 01 bit string to original string
# decompressed_string = decompress_bits_to_string(bit_string)

# # Calculate original data size
# original_size = len(input_string.encode('utf-8'))

# # Calculate compressed data size
# compressed_size = len(bit_string) // 8  # Convert bits to bytes

# # Calculate compression ratio
# compression_ratio = compressed_size / original_size

# # Calculate accuracy
# accuracy = input_string == decompressed_string

# print(f"Original size: {original_size} bytes")
# print(f"Compressed size: {compressed_size} bytes")
# print(f"Compression ratio: {compression_ratio:.2f}")
# print(f"Accuracy: {'Correct' if accuracy else 'Incorrect'}")
# print(f"Decompressed string: {decompressed_string}")