# Script PowerShell para ejecutar optimize_backtest con diferentes configuraciones

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "OPTIMIZE BACKTEST - Menu de Opciones" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Quick test (Random, 20 iteraciones)" -ForegroundColor Yellow
Write-Host "2. Standard (Differential Evolution, 50 iteraciones)" -ForegroundColor Yellow
Write-Host "3. Thorough (Differential Evolution, 100 iteraciones)" -ForegroundColor Yellow
Write-Host "4. Grid Search (Exhaustivo, puede tardar mucho)" -ForegroundColor Yellow
Write-Host "5. Custom (Personalizado)" -ForegroundColor Yellow
Write-Host ""

$choice = Read-Host "Selecciona una opcion (1-5)"

switch ($choice) {
    "1" {
        Write-Host "Ejecutando Quick test..." -ForegroundColor Green
        python optimize_backtest.py --method random --iterations 20
    }
    "2" {
        Write-Host "Ejecutando Standard optimization..." -ForegroundColor Green
        python optimize_backtest.py --method differential_evolution --iterations 50
    }
    "3" {
        Write-Host "Ejecutando Thorough optimization..." -ForegroundColor Green
        python optimize_backtest.py --method differential_evolution --iterations 100
    }
    "4" {
        Write-Host "Ejecutando Grid Search (esto puede tardar mucho tiempo)..." -ForegroundColor Yellow
        python optimize_backtest.py --method grid
    }
    "5" {
        Write-Host ""
        $symbols = Read-Host "Simbolos (ej: BTC/USDT ETH/USDT, o Enter para default)"
        $start = Read-Host "Fecha inicio (YYYY-MM-DD, o Enter para default)"
        $end = Read-Host "Fecha fin (YYYY-MM-DD, o Enter para default)"
        $method = Read-Host "Metodo (grid/random/differential_evolution)"
        $iterations = Read-Host "Iteraciones (o Enter para default)"
        
        $cmd = "python optimize_backtest.py"
        if ($symbols) { $cmd += " --symbols $symbols" }
        if ($start) { $cmd += " --start $start" }
        if ($end) { $cmd += " --end $end" }
        if ($method) { $cmd += " --method $method" }
        if ($iterations) { $cmd += " --iterations $iterations" }
        
        Write-Host "Ejecutando: $cmd" -ForegroundColor Green
        Invoke-Expression $cmd
    }
    default {
        Write-Host "Opcion invalida" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Presiona cualquier tecla para continuar..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")


