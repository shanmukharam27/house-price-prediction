# PowerShell script to run all MLOps pipeline files one by one with live status output
$ErrorActionPreference = "Stop"

$stages = @(
    @{ Name = "Data Loading"; Script = "src\mlops_pipeline\01_load_data.py" },
    @{ Name = "Raw Data Validation"; Script = "src\mlops_pipeline\02_validate_data.py" },
    @{ Name = "Preprocessing"; Script = "src\mlops_pipeline\03_preprocessing.py" },
    @{ Name = "Processed Output Validation"; Script = "src\mlops_pipeline\04_validate_outputs.py" },
    @{ Name = "Training + MLflow + Registry"; Script = "src\mlops_pipeline\05_train_registry.py" },
    @{ Name = "Model Evaluation"; Script = "src\mlops_pipeline\06_model_evaluation.py" },
    @{ Name = "Model Quality Gate"; Script = "src\mlops_pipeline\07_quality_gate.py" },
    @{ Name = "Automated Model Lifecycle"; Script = "src\mlops_pipeline\08_automate_lifecycle.py" },
    @{ Name = "Production Inference"; Script = "src\mlops_pipeline\09_inference.py" }
)

Write-Host "`n========================================================================" -ForegroundColor Cyan
Write-Host "  Starting MLOps Pipeline (Step-by-Step PowerShell Execution)          " -ForegroundColor Cyan
Write-Host "========================================================================`n" -ForegroundColor Cyan

$results = @()
$totalStart = Get-Date

for ($i = 0; $i -lt $stages.Count; $i++) {
    $idx = $i + 1
    $stage = $stages[$i]
    $stepStart = Get-Date

    Write-Host "------------------------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host " [$idx/$($stages.Count)] RUNNING: $($stage.Name)" -ForegroundColor Yellow
    Write-Host " Script: $($stage.Script)" -ForegroundColor DarkGray
    Write-Host "------------------------------------------------------------------------" -ForegroundColor DarkGray

    python $stage.Script
    $exitCode = $LASTEXITCODE
    $duration = [math]::Round(((Get-Date) - $stepStart).TotalSeconds, 2)

    if ($exitCode -eq 0) {
        $status = "PASSED"
        Write-Host "--> [$idx/$($stages.Count)] [PASSED] $($stage.Name) ($($duration)s)`n" -ForegroundColor Green
    } else {
        $status = "FAILED"
        Write-Host "--> [$idx/$($stages.Count)] [FAILED] $($stage.Name) (Exit Code: $exitCode, $($duration)s)`n" -ForegroundColor Red
    }

    $results += [PSCustomObject]@{
        Index    = $idx
        Stage    = $stage.Name
        Script   = (Split-Path $stage.Script -Leaf)
        Duration = "$($duration)s"
        Status   = $status
    }

    if ($exitCode -ne 0) {
        Write-Host "[HALT] Pipeline stopped because stage '$($stage.Name)' failed." -ForegroundColor Red
        break
    }
}

$totalDuration = [math]::Round(((Get-Date) - $totalStart).TotalSeconds, 2)

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "                 STEP-BY-STEP EXECUTION SUMMARY TABLE                   " -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
$results | Format-Table -AutoSize
Write-Host "Total Execution Time: $($totalDuration)s`n" -ForegroundColor Cyan
