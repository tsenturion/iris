<#
.SYNOPSIS
Подготавливает и запускает локальное приложение определения видов ирисов.

.DESCRIPTION
Использует uv.lock при наличии uv, иначе создаёт окружение установленным Python
и устанавливает закреплённые зависимости из requirements.txt. Проверяет активную
модель, при необходимости обучает её на включённом Iris и запускает Streamlit.
Рабочая папка восстанавливается после завершения или ошибки.

.PARAMETER Check
Подготавливает окружение и модель без запуска веб-сервера.

.EXAMPLE
.\start.ps1
Запускает сервер приложения по адресу http://127.0.0.1:8501.

.EXAMPLE
.\start.ps1 -Check
Проверяет готовность окружения и модели без запуска сервера.
#>
[CmdletBinding()]
param([switch]$Check)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
# Переходим к проекту, чтобы относительные пути работали и при запуске скрипта
# из родительской или другой папки; finally возвращает исходное расположение.
Push-Location -LiteralPath $PSScriptRoot
try {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Write-Host 'Подготовка окружения из uv.lock...'
        & uv sync --locked --no-python-downloads
        if ($LASTEXITCODE -ne 0) { throw 'Не удалось подготовить окружение. Нужен установленный Python 3.11 или новее.' }
    }
    else {
        $irisPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $irisPython)) {
            if (Get-Command python -ErrorAction SilentlyContinue) {
                & python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
                if ($LASTEXITCODE -ne 0) { throw 'Установите Python 3.11 или новее либо uv.' }
                & python -m venv .venv
            }
            elseif (Get-Command py -ErrorAction SilentlyContinue) {
                & py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
                if ($LASTEXITCODE -ne 0) { throw 'Установите Python 3.11 или новее либо uv.' }
                & py -3 -m venv .venv
            }
            else { throw 'Установите Python 3.11 или новее либо uv и повторите запуск.' }
            if ($LASTEXITCODE -ne 0) { throw 'Не удалось создать виртуальное окружение.' }
        }
        # Окружение, ранее созданное uv, может не содержать pip. Проверка модулем
        # не пишет ошибку в stderr, который Windows PowerShell может считать сбоем.
        & $irisPython -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pip') else 1)"
        if ($LASTEXITCODE -ne 0) {
            & $irisPython -m ensurepip
            if ($LASTEXITCODE -ne 0) { throw 'Не удалось подготовить pip.' }
        }
        Write-Host 'Установка зависимостей из requirements.txt...'
        & $irisPython -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить зависимости.' }
    }

    $irisPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    Write-Host 'Проверка и подготовка модели...'
    # prepare повторно использует готовую модель, поэтому запуск приложения
    # не обучает новые версии и не сбрасывает выбор пользователя без необходимости.
    & $irisPython -X utf8 -m iris_journal prepare
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось подготовить модель. Подробности находятся в папке logs.' }
    if ($Check) {
        Write-Host 'Проверка завершена. Для открытия приложения запустите .\start.ps1 без -Check.'
    }
    else {
        Write-Host 'Откройте http://127.0.0.1:8501. Для остановки нажмите Ctrl+C.'
        & $irisPython -m streamlit run app.py --server.port 8501
        if ($LASTEXITCODE -ne 0) { throw 'Приложение завершилось с ошибкой. Подробности находятся в папке logs.' }
    }
}
finally {
    Pop-Location
}
