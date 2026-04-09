$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
$pythonBin = if ($env:CHEK_EGO_MINER_PYTHON) { $env:CHEK_EGO_MINER_PYTHON } elseif (Get-Command py -ErrorAction SilentlyContinue) { "py -3" } else { "python" }

$env:PYTHONPATH = "$repoRoot/cli;$repoRoot" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })

if ($pythonBin -eq "py -3") {
  py -3 -m chek_ego_miner.main @args
} else {
  & $pythonBin -m chek_ego_miner.main @args
}
