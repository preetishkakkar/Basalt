param([string]$BuildDirectory = 'build', [ValidateRange(1, 100)][int]$Repeats = 5)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$build = [IO.Path]::GetFullPath((Join-Path $repo $BuildDirectory))
$output = Join-Path $build 'v3.1/performance'
New-Item -ItemType Directory -Force -Path $output | Out-Null
$exe = Join-Path $build 'basalt.exe'
$scene = Join-Path $repo 'tests/data/materials.gltf'
$rows = @()
foreach ($backend in @('gpu', 'cpu-own', 'cpu-embree')) {
    $renderer = if ($backend -eq 'gpu') { 'gpu' } else { 'cpu' }
    $intersector = if ($backend -eq 'cpu-embree') { 'embree' } else { 'own' }
    for ($iteration = 1; $iteration -le $Repeats; ++$iteration) {
        $name = "$backend-$iteration"
        $pfm = Join-Path $output "$name.pfm"
        $arguments = '"{0}" --renderer {1} --intersector {2} --path-temporal --spp 32 --spf 1 --bounces 3 --seed 11 --no-ui --no-vsync --pfm "{3}"' -f $scene, $renderer, $intersector, $pfm
        $process = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -PassThru -Wait
        $log = Join-Path $output "$name.log"
        Copy-Item -LiteralPath (Join-Path $repo 'basalt.log') -Destination $log
        if ($process.ExitCode -ne 0) { throw "$name failed with exit $($process.ExitCode); see $log" }
        if (Select-String -LiteralPath $log -Pattern 'validation:|error:' -Quiet) { throw "$name reported validation/errors; see $log" }
        $metadata = Get-Content -LiteralPath "$pfm.json" -Raw | ConvertFrom-Json
        if ($metadata.spp -ne 32 -or $metadata.preview_scale -ne 1) { throw "$name did not reach 32 full-resolution SPP" }
        $rows += [pscustomobject]@{
            backend = $backend; repeat = $iteration; width = $metadata.width; height = $metadata.height
            filter_gpu_ms = $metadata.reconstruction_gpu_ms; guide_host_copy_ms = $metadata.guide_host_copy_ms
            guide_transfer_gpu_ms = $metadata.guide_transfer_gpu_ms; frame_gpu_ms = $metadata.frame_gpu_ms
            fresh_sample_frame_gpu_ms = $metadata.fresh_sample_frame_gpu_ms
            cpu_paths_per_second = $metadata.cpu_paths_per_second
            frame_wall_delta_ms = $metadata.frame_wall_delta_ms; buffer_bytes = $metadata.reconstruction_buffer_bytes
            frames_rendered = $metadata.frames_rendered; revision = $metadata.source_revision
        }
        Write-Output "${name}: 32 SPP; filter $($metadata.reconstruction_gpu_ms) ms; validation clean"
    }
}
$rows | Export-Csv -NoTypeInformation -Path (Join-Path $output 'measurements.csv')
$summary = foreach ($backend in @('gpu', 'cpu-own', 'cpu-embree')) {
    $group = @($rows | Where-Object backend -eq $backend)
    foreach ($metric in @('filter_gpu_ms', 'guide_host_copy_ms', 'guide_transfer_gpu_ms', 'frame_gpu_ms', 'fresh_sample_frame_gpu_ms', 'cpu_paths_per_second', 'frame_wall_delta_ms', 'buffer_bytes')) {
        $values = @($group | ForEach-Object { [double]$_.$metric } | Sort-Object)
        $mid = [int][Math]::Floor($values.Count / 2)
        $median = if ($values.Count % 2) { $values[$mid] } else { ($values[$mid - 1] + $values[$mid]) / 2 }
        [pscustomobject]@{ backend = $backend; metric = $metric; median = $median; minimum = $values[0]; maximum = $values[-1]; repeats = $values.Count }
    }
}
$summary | Export-Csv -NoTypeInformation -Path (Join-Path $output 'summary.csv')
$summary | Format-Table -AutoSize
