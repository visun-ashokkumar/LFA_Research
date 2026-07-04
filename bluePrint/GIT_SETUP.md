# Git Initialization & GitHub Push Guide

Follow these step-by-step instructions to initialize Git tracking on your local repository and push it to your GitHub account.

## Step 1: Create `.gitkeep` Files (To Track Empty Folders)
Since Git does not track empty folders by default, running this command will place a hidden `.gitkeep` file in each folder so Git preserves your project structure:

```bash
find . -type d -not -path '*/.*' -exec touch {}/.gitkeep \;
```

---

## Step 2: Initialize Git
Initialize a new local Git repository in your root directory:

```bash
git init
```

---

## Step 3: Stage Your Folder Structure
Add all files and the `.gitkeep` placeholders to your staging area:

```bash
git add .
```

---

## Step 4: Create the Initial Commit
Commit your staged structure to your local repository:

```bash
git commit -m "Initial repository structure"
```

---

## Step 5: Rename the Default Branch to `main`
Ensure your main branch is standard-compliant:

```bash
git branch -M main
```

---

## Step 6: Create the Repository on GitHub
1. Go to your web browser and open [GitHub](https://github.com/).
2. Log in and click the **New** repository button (or navigate to [github.com/new](https://github.com/new)).
3. Enter a name for your repository (e.g. `QuantLab` or your renamed folder name).
4. **DO NOT check** "Add a README file", "Add .gitignore", or "Choose a license" options, as you already have the setup locally.
5. Click **Create repository**.

---

## Step 7: Link Your Local Repository to GitHub
Copy the URL of your new GitHub repository (HTTPS or SSH) and add it as the remote target:

```bash
git remote add origin <PASTE_YOUR_GITHUB_REPOSITORY_URL_HERE>
```

*Example:*
```bash
git remote add origin https://github.com/your-username/QuantLab.git
```

---

## Step 8: Push to GitHub
Push your local commit to your GitHub repository:

```bash
git push -u origin main
```
