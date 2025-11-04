; Inno Setup Script for CVVR Server (Windows Installer)
; Builds an installer that deploys the PyInstaller one-folder output
; into Program Files, adds Start Menu/Desktop shortcuts, and a firewall rule.

[Setup]
AppName=CVVR Server
AppVersion=1.0.0
AppId={{7F1A4C0D-8D2C-4F0F-9C0B-9C0DEADBEEF1}
DefaultDirName={pf}\CVVR Server
DefaultGroupName=CVVR Server
OutputBaseFilename=CVVR_Server_Setup_1.0.0
OutputDir=output
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=admin
DisableDirPage=no
DisableProgramGroupPage=no
WizardStyle=modern

[Files]
; Bundle the entire PyInstaller one-folder output directory
; Make sure the relative path points to dist\cvvr_server after build
Source: "..\dist\cvvr_server\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\CVVR Server"; Filename: "{app}\cvvr_server.exe"; WorkingDir: "{app}"
Name: "{commondesktop}\CVVR Server"; Filename: "{app}\cvvr_server.exe"; Tasks: desktopicon; WorkingDir: "{app}"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Run]
; Add inbound firewall rule for the app (port-agnostic; allows the exe)
Filename: "{cmd}"; Parameters: "/C netsh advfirewall firewall add rule name=\"CVVR Server\" dir=in action=allow program=\"{app}\\cvvr_server.exe\" enable=yes"; Flags: runhidden
; Offer to launch after install
Filename: "{app}\\cvvr_server.exe"; Description: "Launch CVVR Server"; Flags: postinstall nowait skipifsilent

[UninstallRun]
; Remove the firewall rule on uninstall (ignore failures)
Filename: "{cmd}"; Parameters: "/C netsh advfirewall firewall delete rule name=\"CVVR Server\""; Flags: runhidden

[Messages]
FinishedLabel=Setup has finished installing {#SetupSetting("AppName")} on your computer.


