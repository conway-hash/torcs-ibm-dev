# scr_client.py
# ---------------------------------------------------------------------------
# Minimal, self-contained SCRC (Simulated Car Racing) UDP client. Adapted from
# the snakeoil3 / torcs_mpc.py clients shipped in ../gym_torcs, stripped of
# command-line parsing and Linux relaunch logic -- TorcsEnv owns the TORCS
# process lifecycle, this object only speaks UDP.
#
#   c = ScrClient(host, port, sensor_angles)
#   c.connect(timeout, attempts)      # SCR (init ...) handshake
#   c.get_servers_input()             # fills c.S.d  (sensor dict)
#   ... set c.R.d['steer'] etc ...
#   c.respond_to_server()             # sends action
#   c.shutdown()
# ---------------------------------------------------------------------------
import socket

DATA_SIZE = 2 ** 17


def _destringify(s):
    if not s:
        return s
    if isinstance(s, str):
        try:
            return float(s)
        except ValueError:
            return s
    if isinstance(s, list):
        if len(s) < 2:
            return _destringify(s[0])
        return [_destringify(i) for i in s]


def _clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class ServerState:
    """Most recent sensor frame from the server, parsed into .d."""

    def __init__(self):
        self.d = {}

    def parse(self, server_string):
        s = server_string.strip()[:-1]
        for item in s.strip().lstrip('(').rstrip(')').split(')('):
            w = item.split(' ')
            self.d[w[0]] = _destringify(w[1:])


class DriverAction:
    """Action to send back to the server."""

    def __init__(self):
        self.d = {'accel': 0.0, 'brake': 0.0, 'clutch': 0.0, 'gear': 1,
                  'steer': 0.0, 'focus': [-90, -45, 0, 45, 90], 'meta': 0}

    def _clip_to_limits(self):
        self.d['steer'] = _clip(self.d['steer'], -1, 1)
        self.d['brake'] = _clip(self.d['brake'], 0, 1)
        self.d['accel'] = _clip(self.d['accel'], 0, 1)
        self.d['clutch'] = _clip(self.d['clutch'], 0, 1)
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d['gear'] = 0
        if self.d['meta'] not in [0, 1]:
            self.d['meta'] = 0

    def __repr__(self):
        self._clip_to_limits()
        out = ''
        for k, v in self.d.items():
            out += '(' + k + ' '
            if isinstance(v, list):
                out += ' '.join(str(x) for x in v)
            else:
                out += '%.3f' % v
            out += ')'
        return out


class ScrClient:
    def __init__(self, host, port, sensor_angles, sid='SCR'):
        self.host = host
        self.port = port
        self.sid = sid
        self.sensor_angles = sensor_angles
        self.so = None
        self.S = ServerState()
        self.R = DriverAction()

    # ---- connection -----------------------------------------------------
    def connect(self, timeout=1.0, attempts=8):
        """Perform the SCR (init ...) handshake. Returns True on success.

        Does NOT relaunch TORCS -- the caller decides whether to kill/relaunch
        the process and retry the handshake.
        """
        self.shutdown()
        self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.so.settimeout(timeout)
        initmsg = '%s(init %s)' % (self.sid, self.sensor_angles)
        for _ in range(max(1, attempts)):
            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error:
                return False
            try:
                data, _ = self.so.recvfrom(DATA_SIZE)
                data = data.decode('utf-8', 'ignore')
            except socket.error:
                continue
            if '***identified***' in data:
                return True
        return False

    # ---- step IO --------------------------------------------------------
    def get_servers_input(self, timeout=1.0, max_wait=8.0):
        """Read one sensor frame into self.S.d.

        Returns one of: 'ok', 'shutdown', 'restart', 'timeout'.
        'timeout' means no parseable frame arrived within max_wait seconds
        (TORCS likely hung) -- caller should relaunch.
        """
        if not self.so:
            return 'timeout'
        import time
        deadline = time.time() + max_wait
        while True:
            try:
                data, _ = self.so.recvfrom(DATA_SIZE)
                data = data.decode('utf-8', 'ignore')
            except socket.error:
                if time.time() > deadline:
                    return 'timeout'
                continue
            if '***identified***' in data:
                continue
            if '***shutdown***' in data:
                return 'shutdown'
            if '***restart***' in data:
                return 'restart'
            if not data:
                if time.time() > deadline:
                    return 'timeout'
                continue
            self.S.parse(data)
            return 'ok'

    def respond_to_server(self):
        if not self.so:
            return
        try:
            self.so.sendto(repr(self.R).encode(), (self.host, self.port))
        except socket.error:
            pass

    def send_restart(self):
        """Ask the server to restart the race (meta=1) for a fresh episode."""
        if not self.so:
            return
        self.R.d['meta'] = 1
        self.respond_to_server()
        self.R.d['meta'] = 0

    def shutdown(self):
        if self.so:
            try:
                self.so.close()
            except socket.error:
                pass
            self.so = None
