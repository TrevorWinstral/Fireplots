import sys
import tinify

def compress_image(uncompressed_image='input.jpg', compressed_image='compressed.jpg'):
	try:	
		if len(sys.argv) > 1:
			uncompressed_image=sys.argv[1]
			compressed_image=sys.argv[2]

		with open('key.txt', 'r') as inp:
			KEY = inp.read()
			tinify.key = KEY

		src = tinify.from_file(uncompressed_image)
		src.to_file(compressed_image)
		return True

	except Exception as e:
		print(e)
		return False

if __name__ == '__main__':
	compress_image()