def delete_corrupt_images(folder):
    for root, dirs, files in os.walk(folder):
        for f in files:
            path = os.path.join(root, f)
            try:
                img = Image.open(path)
                img.verify()
            except:
                print(f"Deleting: {path}")
                os.remove(path)

delete_corrupt_images('data/train')
delete_corrupt_images('data/test')

