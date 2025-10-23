from pathlib import Path

# Check Hindi videos
hindi_dir = Path('data/videos/hindi')
print("Hindi word folders and video counts:")
for word_dir in hindi_dir.iterdir():
    if word_dir.is_dir():
        videos = list(word_dir.glob('*.mp4'))
        print(f"  {word_dir.name}: {len(videos)} videos")

# Check Kannada videos
kannada_dir = Path('data/videos/kannada')
print("\nKannada word folders and video counts:")
for word_dir in kannada_dir.iterdir():
    if word_dir.is_dir():
        videos = list(word_dir.glob('*.mp4'))
        print(f"  {word_dir.name}: {len(videos)} videos")
