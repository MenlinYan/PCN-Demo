# 1. 克隆项目
git clone git@github.com:your-team/private-repo.git
cd private-repo

# 2. 保护主分支，不在上面开发
git checkout main
git pull origin main

# 3. 新建自己的开发分支
git checkout -b feature/my-dev
git push -u origin feature/my-dev

# 4. 在本地开发，提交
git add .
git commit -m "feat: implement xxx"
git push   # 推到远程自己的分支
