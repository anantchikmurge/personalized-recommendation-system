#!/usr/bin/env python3
"""
Script to help push files to GitHub repository
"""

import os
import subprocess
import sys

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_git_status():
    """Check if we're in a git repository"""
    success, stdout, stderr = run_command("git status")
    if not success:
        print("❌ Not in a git repository. Please initialize git first.")
        return False
    return True

def add_files():
    """Add all files to git"""
    print("📁 Adding files to git...")
    success, stdout, stderr = run_command("git add .")
    if success:
        print("✅ Files added successfully!")
        return True
    else:
        print(f"❌ Error adding files: {stderr}")
        return False

def commit_changes():
    """Commit changes with a descriptive message"""
    print("💾 Committing changes...")
    commit_message = "Add Personalized Recommendation System with Streamlit dashboard"
    success, stdout, stderr = run_command(f'git commit -m "{commit_message}"')
    if success:
        print("✅ Changes committed successfully!")
        return True
    else:
        print(f"❌ Error committing changes: {stderr}")
        return False

def push_to_github():
    """Push changes to GitHub"""
    print("🚀 Pushing to GitHub...")
    success, stdout, stderr = run_command("git push origin main")
    if success:
        print("✅ Successfully pushed to GitHub!")
        return True
    else:
        print(f"❌ Error pushing to GitHub: {stderr}")
        return False

def main():
    """Main function to push files to GitHub"""
    print("🚀 PUSHING PERSONALIZED RECOMMENDATION SYSTEM TO GITHUB")
    print("="*60)
    
    # Check git status
    if not check_git_status():
        print("\n📋 To set up git repository:")
        print("1. git init")
        print("2. git remote add origin https://github.com/anantchikmurge/personalized-recommendation-system.git")
        print("3. Run this script again")
        return
    
    # Add files
    if not add_files():
        return
    
    # Commit changes
    if not commit_changes():
        return
    
    # Push to GitHub
    if not push_to_github():
        return
    
    print("\n🎉 SUCCESS! Your Personalized Recommendation System has been pushed to GitHub!")
    print("\n📋 Next steps:")
    print("1. Go to https://github.com/anantchikmurge/personalized-recommendation-system")
    print("2. Verify all files are uploaded")
    print("3. Deploy to Streamlit Cloud:")
    print("   - Go to https://share.streamlit.io")
    print("   - Sign in with GitHub")
    print("   - Create new app")
    print("   - Select your repository")
    print("   - Set main file path to: deploy_app.py")
    print("   - Set requirements file to: requirements-deploy.txt")
    
    print("\n🌐 Your app will be available at: https://your-app-name.streamlit.app")

if __name__ == "__main__":
    main() 