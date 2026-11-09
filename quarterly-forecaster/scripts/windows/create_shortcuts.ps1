# Desktop shortcut + auto-start for Quarterly Forecaster (called by setup.bat).
#   -Mode 1  start when you sign in (Startup-folder shortcut, no admin)       [default]
#   -Mode 2  start at PC boot, signed in or not (Task Scheduler; asks for admin + Windows password)
#   -Mode 3  no auto-start (removes both)
param([string]$Mode = "1")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$startBat = Join-Path $root "start.bat"
$icon = Join-Path $root "static\forecaster.ico"
$taskName = "Quarterly Forecaster"
$startupLink = Join-Path ([Environment]::GetFolderPath("Startup")) "Quarterly Forecaster.lnk"

function New-Shortcut([string]$Path, [string]$Arguments) {
    $shell = New-Object -ComObject WScript.Shell
    $s = $shell.CreateShortcut($Path)
    $s.TargetPath = $startBat
    $s.Arguments = $Arguments
    $s.WorkingDirectory = $root
    $s.IconLocation = $icon
    $s.WindowStyle = 7               # minimised console
    $s.Description = "Quarterly Forecaster"
    $s.Save()
}

if ($Mode -eq "boot-elevated") {
    # Runs elevated (see Mode 2). The task starts the launcher directly (no console window),
    # 1 minute after boot, with no time limit, and restarts it if it fails.
    $user = "$env:USERDOMAIN\$env:USERNAME"
    $cred = Get-Credential -UserName $user -Message "Your Windows password (lets it run at boot before you sign in)"
    $action = New-ScheduledTaskAction -Execute (Join-Path $root ".venv\Scripts\pythonw.exe") `
        -Argument "-m forecaster.launcher" -WorkingDirectory $root
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $trigger.Delay = "PT1M"
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable `
        -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 1) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
        -User $cred.UserName -Password $cred.GetNetworkCredential().Password -RunLevel Limited -Force | Out-Null
    Write-Host "Scheduled: starts at every boot."
    Start-Sleep -Seconds 3
    exit 0
}

New-Shortcut (Join-Path ([Environment]::GetFolderPath("Desktop")) "Quarterly Forecaster.lnk") ""
Write-Host "Desktop shortcut created."
Remove-Item $startupLink -ErrorAction SilentlyContinue

if ($Mode -eq "2") {
    Write-Host "Windows will ask for admin rights, then your Windows password."
    Start-Process powershell -Verb RunAs -Wait -ArgumentList `
        "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Mode boot-elevated"
    if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
        Write-Host "It will start at PC boot (1 minute after Windows starts)."
    } else {
        Write-Host "Boot task not created; falling back to start at sign-in."
        New-Shortcut $startupLink "/boot"
    }
} elseif ($Mode -eq "3") {
    Write-Host "Auto-start off."
} else {
    New-Shortcut $startupLink "/boot"
    Write-Host "It will start automatically when you sign in to Windows."
}
if ($Mode -ne "2" -and (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue)) {
    Write-Host "Removing the old boot task (needs admin)..."
    Start-Process schtasks -Verb RunAs -Wait -ArgumentList "/Delete /F /TN `"$taskName`""
}
