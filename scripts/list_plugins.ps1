# Windows に入っているプラグイン（VST3／VST2／CLAP）と Native Instruments 製品の一覧をテキストに書く。
# 使い方: PowerShell で  powershell -ExecutionPolicy Bypass -File scripts\list_plugins.ps1
#        別のフォルダも見るなら  powershell -ExecutionPolicy Bypass -File scripts\list_plugins.ps1 -Extra "E:\VST","F:\Plugins"
# 出力: デスクトップの plugins.txt（パスと更新日だけ。音声や設定には触れない）

param([string[]]$Extra = @())

$out = Join-Path ([Environment]::GetFolderPath("Desktop")) "plugins.txt"
$lines = @()
$lines += "# プラグイン一覧 " + (Get-Date -Format "yyyy-MM-dd HH:mm")
$lines += ""

$dirs = @(
    "D:\Music\Plugin",  # Takahiro のプラグインの置き場（2026-10-25）
    "$env:ProgramFiles\Common Files\VST3",
    "$env:ProgramFiles\Common Files\CLAP",
    "$env:ProgramFiles\Common Files\VST2",
    "$env:ProgramFiles\VSTPlugins",
    "$env:ProgramFiles\Steinberg\VSTPlugins",
    "$env:ProgramFiles\Native Instruments\VSTPlugins 64 bit",
    "${env:ProgramFiles(x86)}\VSTPlugins",
    "${env:ProgramFiles(x86)}\Steinberg\VSTPlugins",
    "$env:LOCALAPPDATA\Programs\Common\VST3",
    "$env:LOCALAPPDATA\Programs\Common\CLAP"
) + $Extra

foreach ($d in $dirs) {
    if (-not (Test-Path $d)) { continue }
    $lines += "## $d"
    Get-ChildItem -Path $d -Recurse -Include *.vst3, *.dll, *.clap -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch "\\Contents\\(Resources|x86_64-win)\\" -or $_.Extension -eq ".vst3" } |
        Sort-Object FullName | ForEach-Object {
            $rel = $_.FullName.Substring($d.Length).TrimStart("\")
            $lines += ("- " + $rel + "  (" + $_.LastWriteTime.ToString("yyyy-MM-dd") + ")")
        }
    $lines += ""
}

# Native Instruments 製品（レジストリ）
$ni = "HKLM:\SOFTWARE\Native Instruments"
if (Test-Path $ni) {
    $lines += "## Native Instruments 製品（レジストリ）"
    Get-ChildItem $ni -ErrorAction SilentlyContinue | ForEach-Object {
        $name = $_.PSChildName
        $content = (Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue).ContentDir
        if ($content) { $lines += "- $name  ($content)" } else { $lines += "- $name" }
    }
    $lines += ""
}

# インストール済みの音楽系ソフト（名前だけ）
$lines += "## インストール済みのソフト（名前に音楽系の語を含むもの）"
$keys = @("HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*", "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*", "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*")
$pattern = "Native Instruments|Komplete|Maschine|Kontakt|Steinberg|Cubase|SpectraLayers|Serato|Arturia|iZotope|FabFilter|Waves|Spectrasonics|Toontrack|XLN|Addictive|Superior|EZdrummer|Ableton|FL Studio|Image-Line|Studio One|PreSonus|Reaper|Bitwig|Plugin Alliance|Softube|u-he|Valhalla|Output|Spitfire|IK Multimedia|Roland|Korg|Yamaha|Vienna|Melda|Kilohearts|Xfer|Serum|Vital|Pigments|Omnisphere|Keyscape|BFD|GetGood|Slate|Celemony|Melodyne|Sonarworks|Antares|Soundtoys|Eventide|Lexicon|Acustica|Audio Damage|Baby Audio|Cableguys|Sugar Bytes|Native"
Get-ItemProperty $keys -ErrorAction SilentlyContinue |
    Where-Object { $_.DisplayName -and $_.DisplayName -match $pattern } |
    Sort-Object DisplayName -Unique | ForEach-Object {
        $lines += ("- " + $_.DisplayName + "  " + $_.DisplayVersion)
    }

$lines | Set-Content -Path $out -Encoding UTF8
Write-Host "書き出した: $out  ($($lines.Count) 行)"
