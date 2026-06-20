@echo off
chcp 65001 > nul
echo 正在启动 MBTI 精神病院...
echo 请勿关闭此窗口，关闭后服务将停止。
echo.
echo 启动成功后，请在浏览器访问: http://localhost:5001
echo 手机访问请使用: http://[你的局域网IP]:5001
echo.
python app.py
pause