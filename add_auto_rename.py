"""
Script to add auto-rename feature to train_gui_with_recording.py
This adds the functionality to automatically rename models based on language
"""

def add_auto_rename():
    file_path = './train_gui_with_recording.py'
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already added
    if '_auto_rename_model_files' in content:
        print("✓ Auto-rename feature already present!")
        return
    
    # 1. Add the method call after training completion
    old_completion = '''            self.log("="*60)
            self.log("TRAINING COMPLETED SUCCESSFULLY")
            self.log("="*60)
            
            messagebox.showinfo("Success", "Training completed successfully!")'''
    
    new_completion = '''            self.log("="*60)
            self.log("TRAINING COMPLETED SUCCESSFULLY")
            self.log("="*60)
            
            # Auto-rename model based on trained languages
            self._auto_rename_model_files()
            
            messagebox.showinfo("Success", "Training completed successfully!")'''
    
    if old_completion in content:
        content = content.replace(old_completion, new_completion)
        print("✓ Added auto-rename call after training completion")
    else:
        print("✗ Could not find training completion section")
        return
    
    # 2. Add the auto-rename method
    old_log_method = '''    def log(self, message):
        """Log message to console"""
        self.console.insert(tk.END, message + "\\n")
        self.console.see(tk.END)
        self.root.update_idletasks()'''
    
    new_methods = '''    def _auto_rename_model_files(self):
        """Automatically rename model and class_mapping based on trained languages (no backup)"""
        try:
            import shutil
            import os
            
            # Detect which languages were trained
            video_dir = Path('./data/videos')
            if not video_dir.exists():
                return
            
            # Get list of languages in training data
            languages = []
            for item in video_dir.iterdir():
                if item.is_dir():
                    lang_name = item.name.lower()
                    if lang_name in ['hindi', 'kannada', 'english', 'tamil', 'telugu']:
                        # Check if this language has data
                        if any(item.iterdir()):
                            languages.append(lang_name)
            
            if not languages:
                self.log("⚠ No language folders detected, keeping default names")
                return
            
            # Sort for consistent naming
            languages.sort()
            
            # Determine output filename
            if len(languages) == 1:
                # Single language model
                lang_suffix = languages[0]
                model_name = f"best_model_{lang_suffix}.h5"
                mapping_name = f"class_mapping_{lang_suffix}.json"
                self.log(f"\\n🔄 Single language detected: {languages[0].upper()}")
            else:
                # Multi-language model
                model_name = f"best_model_multi.h5"
                mapping_name = f"class_mapping_multi.json"
                self.log(f"\\n🔄 Multiple languages detected: {', '.join([l.upper() for l in languages])}")
            
            # Rename model file (overwrite if exists)
            old_model = Path('./models/best_model.h5')
            new_model = Path(f'./models/{model_name}')
            
            if old_model.exists():
                # Delete existing target file if it exists (no backup)
                if new_model.exists():
                    os.remove(str(new_model))
                    self.log(f"   🗑️  Removed old: {model_name}")
                
                shutil.move(str(old_model), str(new_model))
                self.log(f"   ✅ Model saved: {model_name}")
            
            # Rename class mapping file (overwrite if exists)
            old_mapping = Path('./models/class_mapping.json')
            new_mapping = Path(f'./models/{mapping_name}')
            
            if old_mapping.exists():
                # Delete existing target file if it exists (no backup)
                if new_mapping.exists():
                    os.remove(str(new_mapping))
                    self.log(f"   🗑️  Removed old: {mapping_name}")
                
                shutil.move(str(old_mapping), str(new_mapping))
                self.log(f"   ✅ Mapping saved: {mapping_name}")
            
            self.log(f"\\n💡 Ready to use: {model_name}")
            if len(languages) == 1:
                self.log(f"   For auto-detection, train other languages separately")
            
        except Exception as e:
            self.log(f"\\n⚠ Auto-rename failed: {e}")
            self.log("   Files saved as: best_model.h5 and class_mapping.json")
    
    def log(self, message):
        """Log message to console"""
        self.console.insert(tk.END, message + "\\n")
        self.console.see(tk.END)
        self.root.update_idletasks()'''
    
    if old_log_method in content:
        content = content.replace(old_log_method, new_methods)
        print("✓ Added _auto_rename_model_files method")
    else:
        print("✗ Could not find log method")
        return
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✅ Auto-rename feature added successfully!")
    print("\nNow when you train:")
    print("  - Hindi only → best_model_hindi.h5")
    print("  - Kannada only → best_model_kannada.h5")
    print("  - Both → best_model_multi.h5")

if __name__ == "__main__":
    add_auto_rename()
