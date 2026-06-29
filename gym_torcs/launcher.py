# launcher.py — hybrid_learning
# Adapted from gym_torcs_learning/launcher.py; uses hybrid CFG.
import os
import shutil
import socket
import subprocess
import time

from config import CFG

_RACE_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE params SYSTEM "../params.dtd">
<params name="Quick Race">
 <section name="Header">
  <attstr name="name" val="Quick Race"/>
  <attstr name="description" val="Hybrid SAC training race"/>
  <attnum name="priority" val="10"/>
 </section>
 <section name="Tracks">
  <attnum name="maximum number" val="1"/>
  <section name="1">
   <attstr name="name" val="{track}"/>
   <attstr name="category" val="{category}"/>
  </section>
 </section>
 <section name="Races">
  <section name="1"><attstr name="name" val="Quick Race"/></section>
 </section>
 <section name="Quick Race">
  <attnum name="distance" unit="km" val="0"/>
  <attstr name="type" val="race"/>
  <attstr name="starting order" val="drivers list"/>
  <attstr name="restart" val="yes"/>
  <attnum name="laps" val="{laps}"/>
  <section name="Starting Grid">
   <attnum name="rows" val="1"/>
   <attnum name="distance to start" val="25"/>
   <attnum name="initial speed" val="0"/>
   <attnum name="initial height" val="0.2"/>
  </section>
 </section>
 <section name="Drivers">
  <attnum name="maximum number" val="1"/>
  <attstr name="focused module" val="scr_server"/>
  <attnum name="focused idx" val="{idx}"/>
  <section name="1">
   <attnum name="idx" val="{idx}"/>
   <attstr name="module" val="scr_server"/>
  </section>
 </section>
</params>
"""


def write_race_config(idx, cfg=CFG):
    os.makedirs(cfg.raceconfig_dir, exist_ok=True)
    path = os.path.join(cfg.raceconfig_dir,
                        "%s%d.xml" % (cfg.raceconfig_prefix, idx))
    xml = _RACE_TEMPLATE.format(track=cfg.track_name,
                                category=cfg.track_category,
                                laps=cfg.laps_per_race, idx=idx)
    with open(path, "w") as f:
        f.write(xml)
    return path


def handshake_ok(port, host=CFG.host, sensor_angles=CFG.track_sensor_angles,
                 timeout=1.0, attempts=8):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    msg = ("SCR(init %s)" % sensor_angles).encode()
    try:
        for _ in range(attempts):
            try:
                s.sendto(msg, (host, port))
                data, _ = s.recvfrom(2 ** 16)
                if b'identified' in data:
                    return True
            except socket.error:
                continue
    finally:
        s.close()
    return False


def ensure_car(cfg=CFG, verbose=True):
    """Self-heal: force every scr_server slot to cfg.car_name (and deploy the
    matching setup), so all workers + eval always run the SAME car even if
    something (e.g. a git checkout) reverts the TORCS config. No-op if already
    correct. Returns True if it had to fix anything."""
    import re
    sdir = os.path.join(cfg.torcs_dir, "drivers", "scr_server")
    xml = os.path.join(sdir, "scr_server.xml")
    try:
        t = open(xml).read()
    except Exception:
        return False
    cars = re.findall(r'car name" val="([^"]+)"', t)
    ok = bool(cars) and all(c == cfg.car_name for c in cars)
    if ok:
        return False
    if not os.path.exists(xml + ".rt_bak"):
        shutil.copy(xml, xml + ".rt_bak")
    t = re.sub(r'(car name" val=")[^"]+(")',
               r"\g<1>" + cfg.car_name + r"\g<2>", t)
    open(xml, "w").write(t)
    # deploy the matching car setup to every index dir (if we have a canonical)
    if cfg.car_setup_src and os.path.exists(cfg.car_setup_src):
        for i in range(10):
            dst = os.path.join(sdir, str(i), "default.xml")
            if os.path.isdir(os.path.dirname(dst)):
                try:
                    shutil.copy(cfg.car_setup_src, dst)
                except Exception:
                    pass
    if verbose:
        print("[car] healed scr_server -> %s (was mixed/%s)"
              % (cfg.car_name, set(cars)), flush=True)
    return True


def kill_all_torcs():
    try:
        subprocess.run(["taskkill", "/F", "/IM", "wtorcs.exe"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


class TorcsInstance:
    def __init__(self, idx, cfg=CFG, gui=False):
        self.idx = idx
        self.cfg = cfg
        self.gui = gui
        self.port = cfg.port_for(idx)
        self.proc = None
        self.cfg_path = write_race_config(idx, cfg)
        self.relaunches = 0

    def _spawn(self):
        flags = list(self.cfg.torcs_flags)
        if self.gui:
            flags = [f for f in flags if f != "-T"]
        args = [self.cfg.torcs_exe] + flags + ["-r", self.cfg_path]
        cf = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.proc = subprocess.Popen(
            args, cwd=self.cfg.torcs_dir,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=cf)

    def spawn_menu(self):
        """Launch TORCS showing its GUI menu (no -r flag).
        The graphics module loads from the menu, so the race window is visible.
        Use with eval.py --gui: user starts the race manually, Python attaches."""
        flags = [f for f in self.cfg.torcs_flags if f != "-T"]
        args = [self.cfg.torcs_exe] + flags
        cf = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
        self.proc = subprocess.Popen(args, cwd=self.cfg.torcs_dir,
                                     creationflags=cf)

    def kill(self):
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass
            self.proc = None

    def is_alive(self):
        return self.proc is not None and self.proc.poll() is None

    def launch(self, wait_handshake=True):
        for attempt in range(self.cfg.max_relaunch_tries):
            self.kill()
            self._spawn()
            if not wait_handshake:
                return True
            time.sleep(2.0)
            if handshake_ok(self.port, self.cfg.host,
                            self.cfg.track_sensor_angles,
                            self.cfg.handshake_timeout_s,
                            self.cfg.handshake_attempts):
                return True
            self.relaunches += 1
            print("[launcher] idx %d port %d no handshake (try %d/%d)"
                  % (self.idx, self.port, attempt + 1,
                     self.cfg.max_relaunch_tries))
        raise RuntimeError("TORCS idx %d failed to start" % self.idx)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="TORCS launch utilities")
    ap.add_argument("--kill", action="store_true",
                    help="kill every wtorcs.exe (clear strays/zombies)")
    ap.add_argument("--smoke", type=int, metavar="N",
                    help="launch N headless TORCS, confirm handshake, exit")
    ap.add_argument("--gui-race", action="store_true",
                    help="launch a VISIBLE corkscrew+scr_server race (port 3001) "
                         "and leave it running, for `eval.py --attach`")
    a = ap.parse_args()
    if a.kill:
        kill_all_torcs()
        print("killed all wtorcs.exe")
    elif a.gui_race:
        kill_all_torcs(); time.sleep(1.0)
        cfg_path = write_race_config(0, CFG)
        inst = TorcsInstance(0, CFG, gui=True)
        inst._spawn()                       # launch, do NOT block / kill on exit
        print("=" * 64)
        print(" TORCS GUI race launching on port %d (pid %d)."
              % (CFG.port_for(0), inst.proc.pid))
        print(" Wait for the window (~10 s); the car sits at the grid.")
        print(" Then, in another terminal:   python eval.py --attach")
        print(" Close TORCS later with:      python launcher.py --kill")
        print("")
        print(" If NO window appears, run this command yourself to see errors:")
        print('   "%s" -nofuel -nodamage -nolaptime -r "%s"'
              % (CFG.torcs_exe, cfg_path))
        print("=" * 64)
        # exit without killing — the GUI race keeps running for --attach
    elif a.smoke:
        kill_all_torcs(); time.sleep(1.0)
        insts = []
        try:
            for i in range(a.smoke):
                inst = TorcsInstance(i, CFG, gui=False)
                inst.launch(wait_handshake=False)
                insts.append(inst)
                time.sleep(2.0)
                ok = handshake_ok(CFG.port_for(i))
                print("idx %d port %d : %s" % (i, CFG.port_for(i),
                                               "OK" if ok else "NO HANDSHAKE"))
        finally:
            for inst in insts:
                inst.kill()
            kill_all_torcs()
    else:
        ap.print_help()
