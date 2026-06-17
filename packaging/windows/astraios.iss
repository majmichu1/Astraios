; Astraios — Windows installer (Inno Setup).
;
; Ships small (~5 MB: the app wheel + the uv installer). At the end of install
; it runs bootstrap.ps1, which detects the GPU and downloads the matching
; PyTorch (CUDA or CPU) plus dependencies onto the user's machine. This is what
; makes real GPU acceleration possible without a multi-GB download or hitting
; GitHub's 2 GB release-asset limit.
;
; Per-user install (no admin / UAC) so the venv and the ~2.5 GB CUDA download
; land in the user's profile.

#define MyAppName "Astraios"
#ifndef MyAppVersion
  #define MyAppVersion "0.1.0"
#endif
#define MyAppPublisher "Astraios Contributors"
#define MyAppURL "https://github.com/majmichu1/Astraios"

[Setup]
AppId={{A5712A05-7E4C-4D2B-9F1A-AB0C0FFEE001}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={localappdata}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=installer_output
OutputBaseFilename=Astraios-Setup-{#MyAppVersion}
SetupIconFile=astraios.ico
UninstallDisplayIcon={app}\astraios.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "astraios.ico";  DestDir: "{app}"; Flags: ignoreversion
Source: "uv.exe";        DestDir: "{app}"; Flags: ignoreversion
Source: "bootstrap.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "*.whl";         DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Astraios"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m astraios"; WorkingDir: "{app}"; IconFilename: "{app}\astraios.ico"
Name: "{group}\Uninstall Astraios"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Astraios"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m astraios"; WorkingDir: "{app}"; IconFilename: "{app}\astraios.ico"; Tasks: desktopicon

[Run]
; The big step: build the env + download the right PyTorch. Visible so the user
; sees progress (it can take several minutes on the first install).
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\bootstrap.ps1"" -AppDir ""{app}"""; \
  StatusMsg: "Setting up Python and downloading PyTorch for your GPU (several minutes)..."; \
  Flags: waituntilterminated
Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m astraios"; WorkingDir: "{app}"; \
  Description: "{cm:LaunchProgram,Astraios}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}"
