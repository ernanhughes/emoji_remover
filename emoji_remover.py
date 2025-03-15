import os
import emoji
import shutil
from datetime import datetime

# Backup folder path
BACKUP_DIR = r"C:\backup"

def extract_emojis(text):
    """Extract all emojis from a given text string."""
    emoji_list = [char for char in text if char in emoji.EMOJI_DATA]
    return emoji_list

def remove_emojis(text):
    """Remove all emojis from a text string."""
    return ''.join(char for char in text if char not in emoji.EMOJI_DATA)

def create_backup(file_path, backup_folder):
    """Backup original file before modification."""
    os.makedirs(backup_folder, exist_ok=True)
    backup_file_path = os.path.join(backup_folder, os.path.basename(file_path))
    shutil.copy2(file_path, backup_file_path)

def process_folder(folder_path):
    """Scan all text-based files in a folder, backup, detect, and remove emojis."""
    today_date = datetime.now().strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
    backup_folder = os.path.join(BACKUP_DIR, today_date)

    emoji_results = {}

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)

            # Process only text-based files
            if file.endswith((".txt", ".md", ".json", ".py", ".log")):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    emojis = extract_emojis(content)

                if emojis:
                    emoji_results[file_path] = emojis

                    # Backup the original file
                    create_backup(file_path, backup_folder)

                    # Remove emojis and overwrite the file
                    cleaned_text = remove_emojis(content)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(cleaned_text)

    return emoji_results, backup_folder

if __name__ == "__main__":
    folder_path = input("Enter folder path: ").strip()
    
    if not os.path.exists(folder_path):
        print("❌ Folder not found. Please enter a valid path.")
    else:
        results, backup_path = process_folder(folder_path)

        if results:
            print("\n📊 Emoji Removal Report:")
            for file, emojis in results.items():
                print(f"📂 File: {file}")
                print(f"🔍 Removed Emojis: {''.join(set(emojis))}\n")

            print(f"✅ All emojis have been removed from detected files!")
            print(f"📂 Original files have been backed up in: {backup_path}")
        else:
            print("✅ No emojis found in any files!")
