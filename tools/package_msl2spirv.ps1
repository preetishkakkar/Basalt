# Packages the compiler as the release asset CMake fetches, and prints the
# SHA-256 to paste into BASALT_MSL2SPIRV_SHA256 in cmake/msl2spirv.cmake.
#
#   powershell -File tools/package_msl2spirv.ps1
#   gh release create msl2spirv-1.0.0 build/dist/msl2spirv-1.0.0-windows-x64.zip
[CmdletBinding()]
param(
  [string]$Version = '1.0.0',
  [string]$OutputDirectory = ''
)

$ErrorActionPreference = 'Stop'
# $PSScriptRoot is not bound yet where the parameter defaults are evaluated.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($OutputDirectory -eq '') { $OutputDirectory = Join-Path $here '../build/dist' }
$executable = Join-Path $here 'metal2vulkan/bin/msl2spirv.exe'
if (-not (Test-Path $executable)) { throw "No compiler at $executable" }

$reported = (& $executable --version | Select-Object -First 1)
if ($reported -notmatch [regex]::Escape($Version)) {
  throw "The compiler reports '$reported', which is not version $Version"
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$archive = Join-Path $OutputDirectory "msl2spirv-$Version-windows-x64.zip"
if (Test-Path $archive) { Remove-Item $archive }
Compress-Archive -Path $executable -DestinationPath $archive -CompressionLevel Optimal

$hash = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLower()
(Resolve-Path $archive).Path
"$([math]::Round((Get-Item $archive).Length / 1MB, 1)) MB"
"SHA256=$hash"
