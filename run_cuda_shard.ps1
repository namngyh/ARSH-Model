param(
    [Parameter(Mandatory=$true)][string]$Output,
    [Parameter(Mandatory=$true)][int[]]$Folds,
    [int[]]$Horizons = @(1,2,3,5,10,15,20,30,45,60,90),
    [int[]]$KValues = @(2,3,4,5,6,7),
    [ValidateSet('daily_sequence','session_sequence')][string]$Policy = 'session_sequence',
    [ValidateSet('raw','intraday_adjusted')][string]$ReturnVariant = 'raw',
    [int]$TrainYears = 3,
    [switch]$ExpandingWindow,
    [switch]$Overlapping,
    [switch]$ExcludeZero,
    [ValidateSet('keep','exclude_train_5sigma')][string]$OutlierPolicy = 'keep',
    [switch]$ExcludeRolloverAdjacent,
    [double]$TimeBudgetMinutes = 0
)
$ErrorActionPreference = 'Stop'
$project=$PSScriptRoot
$python=Join-Path $project '.venv-cuda\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Chạy setup_cuda.ps1 trước.' }
$argsList=@((Join-Path $project 'arsh_v05.py'),'--data',(Join-Path $project 'data\ohlc_export.csv'),
    '--output',$Output,'--backend','cuda','--device','cuda:0','--policy',$Policy,
    '--return-variant',$ReturnVariant,'--train-years',$TrainYears,'--outlier-policy',$OutlierPolicy,
    '--horizons')+$Horizons+@('--k-values')+$KValues+@('--folds')+$Folds+@('--resume')
if($ExpandingWindow){$argsList+='--expanding-window'}
if($Overlapping){$argsList+='--overlapping'}
if($ExcludeZero){$argsList+='--exclude-zero'}
if($ExcludeRolloverAdjacent){$argsList+='--exclude-rollover-adjacent'}
if($TimeBudgetMinutes -gt 0){$argsList+=@('--time-budget-minutes',$TimeBudgetMinutes)}
& $python @argsList
if($LASTEXITCODE -ne 0){throw "CUDA shard failed with exit code $LASTEXITCODE"}
