param(
    [string]$PythonExe = "python",
    [string]$Result2Q = "results/twobody_paper_final_20260409_170957",
    [string]$Result3Q = "results/twobody_paper_final_3q_20260410_120037",
    [string]$Result4Q = "results/twobody_paper_final_4q_20260410_122523",
    [string]$Result5Q = "results/twobody_paper_final_5q_20260416_205449",
    [string]$Result6Q = "",
    [string]$Result8Q = "",
    [switch]$ResumeExisting,
    [switch]$Skip6Q,
    [switch]$Skip8Q
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RequiredSummaryFiles = @(
    "transition_summary.csv",
    "component_sweep_summary.csv",
    "composite_sweep_summary.csv",
    "fingerprint_noise_sweep_summary.csv",
    "fingerprint_distance_summary.csv",
    "fingerprint_tamper_sweep_summary.csv"
)

function Test-CompletedPaperResult {
    param(
        [string]$Path
    )

    if (-not $Path) {
        return $false
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        return $false
    }

    foreach ($file in $RequiredSummaryFiles) {
        if (-not (Test-Path -LiteralPath (Join-Path $Path $file) -PathType Leaf)) {
            return $false
        }
    }
    return $true
}

function Find-LatestCompletedPaperResult {
    param(
        [string]$Prefix
    )

    $matches = Get-ChildItem "results" -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "$Prefix*" } |
        Sort-Object LastWriteTime -Descending

    foreach ($match in $matches) {
        if (Test-CompletedPaperResult -Path $match.FullName) {
            return $match.FullName
        }
    }
    return $null
}

function Invoke-PaperRun {
    param(
        [string]$ConfigPath
    )

    $output = @()
    & $PythonExe "scripts/run_twobody_paper_figures.py" --config $ConfigPath | Tee-Object -Variable output
    if ($LASTEXITCODE -ne 0) {
        throw "Paper run failed for $ConfigPath (exit code $LASTEXITCODE)"
    }
    $outputDir = ($output | Select-String '^output_dir=' | Select-Object -Last 1).Line
    if (-not $outputDir) {
        throw "Could not parse output_dir from runner output for $ConfigPath"
    }
    return ($outputDir -replace '^output_dir=', '').Trim()
}

$result6Q = $Result6Q
$result8Q = $Result8Q

if ($ResumeExisting) {
    if (-not (Test-CompletedPaperResult -Path $result6Q)) {
        $result6Q = Find-LatestCompletedPaperResult -Prefix "twobody_paper_final_6q_"
    }
    if (-not (Test-CompletedPaperResult -Path $result8Q)) {
        $result8Q = Find-LatestCompletedPaperResult -Prefix "twobody_paper_final_8q_"
    }
}

if ($result6Q) {
    Write-Host "Using existing 6Q result: $result6Q"
} elseif (-not $Skip6Q) {
    $result6Q = Invoke-PaperRun -ConfigPath "configs/twobody/paper_figures_final_6q.yaml"
}

if ($result8Q) {
    Write-Host "Using existing 8Q result: $result8Q"
} elseif (-not $Skip8Q) {
    $result8Q = Invoke-PaperRun -ConfigPath "configs/twobody/paper_figures_final_8q.yaml"
}

$resultDirArgs = @(
    "--result-dir", "2=$Result2Q",
    "--result-dir", "3=$Result3Q",
    "--result-dir", "4=$Result4Q",
    "--result-dir", "5=$Result5Q"
)

if ($result6Q) {
    $resultDirArgs += @("--result-dir", "6=$result6Q")
}

if ($result8Q) {
    $resultDirArgs += @("--result-dir", "8=$result8Q")
}

$analysisOutput = @()
& $PythonExe "scripts/analyze_twobody_scaling.py" @resultDirArgs | Tee-Object -Variable analysisOutput
if ($LASTEXITCODE -ne 0) {
    throw "Scaling analysis failed (exit code $LASTEXITCODE)"
}

$plotOutput = @()
& $PythonExe "scripts/plot_twobody_scalability_fullscale.py" @resultDirArgs | Tee-Object -Variable plotOutput
if ($LASTEXITCODE -ne 0) {
    throw "Fullscale plot generation failed (exit code $LASTEXITCODE)"
}
