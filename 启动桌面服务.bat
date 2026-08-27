@echo off
chcp 65001 >nul
title InfinityScale 本地图像处理工具

echo ===================================================
echo   InfinityScale 本地图像放大与矢量追踪工具
echo ===================================================
echo 正在启动本地服务（端口 8080）...

where py >nul 2>nul
if %errorlevel% equ 0 (
    start "" /b py -m http.server 8080
    timeout /t 2 /nobreak >nul
    start "" http://localhost:8080
    goto end
)

where python >nul 2>nul
if %errorlevel% equ 0 (
    start "" /b python -m http.server 8080
    timeout /t 2 /nobreak >nul
    start "" http://localhost:8080
    goto end
)

where node >nul 2>nul
if %errorlevel% equ 0 (
    echo [提示] 未检测到 Python，将通过 npx 启动 http-server；首次可能需要联网下载。
    start "" /b cmd /c "npx --yes http-server -p 8080 -s"
    timeout /t 3 /nobreak >nul
    start "" http://localhost:8080
    goto end
)

echo ---------------------------------------------------
echo [提示] 未检测到 Python 或 Node.js，改为直接打开 index.html。
echo file 模式不能安装为 PWA；AI、矢量和 PDF 依赖首次仍需联网载入。
echo 已成功缓存的资源可在后续离线使用，但不保证所有模型已缓存。
echo ---------------------------------------------------
pause
start "" index.html
goto done

:end
echo [完成] 本地页面已打开：http://localhost:8080
echo AI 模型首次运行需要联网下载；是否命中缓存请以浏览器网络日志为准。

:done
timeout /t 3 /nobreak >nul
exit
