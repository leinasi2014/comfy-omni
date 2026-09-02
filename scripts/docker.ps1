[CmdletBinding()]
param(
    [ValidateSet("docs", "quality", "package", "image", "cli")]
    [string] $Action = "quality",
    [ValidateSet("3.10", "3.11", "3.12", "3.13")]
    [string] $PythonVersion = "3.13",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $CliArguments
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repositoryRoot
try {
    $sourceCommit = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $sourceCommit -notmatch "^[0-9a-f]{40}$") {
        throw "Cannot resolve the exact lowercase Git commit."
    }
    $sourceDirty = if ((& git status --porcelain --untracked-files=normal)) { "1" } else { "0" }
    $target = @{
        docs = "documentation"
        quality = "quality"
        package = "package-check"
        image = "runtime"
        cli = "runtime"
    }[$Action]
    $image = "comfy-omni:$Action-$($sourceCommit.Substring(0, 12))-py$PythonVersion"

    & docker build `
        --build-arg "PYTHON_VERSION=$PythonVersion" `
        --build-arg "COMFY_OMNI_BUILD_COMMIT=$sourceCommit" `
        --build-arg "COMFY_OMNI_BUILD_DIRTY=$sourceDirty" `
        --target $target `
        --tag $image `
        .
    if ($LASTEXITCODE -ne 0) {
        throw "Docker build failed with exit code $LASTEXITCODE."
    }

    if ($Action -eq "cli") {
        if (-not $CliArguments) {
            $CliArguments = @("--help")
        }
        & docker run --rm `
            --network none `
            --read-only `
            --tmpfs "/tmp:rw,noexec,nosuid,size=64m" `
            --cap-drop ALL `
            --security-opt no-new-privileges `
            $image @CliArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Containerized CLI failed with exit code $LASTEXITCODE."
        }
    }
}
finally {
    Pop-Location
}
