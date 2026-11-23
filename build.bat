@echo off
setlocal
cd /d "%~dp0"

set "VCVARS="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%I"
)
if defined VSINSTALL if exist "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo vcvars64.bat bulunamadi: %VCVARS%
    exit /b 1
)

call "%VCVARS%"
if errorlevel 1 exit /b 1

if not exist build mkdir build

set "CLFLAGS=/nologo /O2 /openmp /EHsc /utf-8 /std:c++17 /I src"
cl %CLFLAGS% /arch:AVX2 /c /Fo:build\matrix.obj src\matrix.cpp || exit /b 1
cl %CLFLAGS% /arch:AVX2 /c /Fo:build\qlora_avx2.obj src\qlora_avx2.cpp || exit /b 1
cl %CLFLAGS% /arch:AVX512 /c /Fo:build\qlora_vnni.obj src\qlora_vnni.cpp || exit /b 1
cl %CLFLAGS% /arch:AVX2 /c /Fo:build\main.obj src\main.cpp || exit /b 1
cl /nologo /Fe:build\minicore.exe build\matrix.obj build\qlora_avx2.obj build\main.obj build\qlora_vnni.obj /link psapi.lib || exit /b 1

echo LINK_OK  build\minicore.exe
exit /b 0
