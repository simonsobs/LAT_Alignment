"""
Functions for integrating with the Atlas Copco IxB tool
"""

import argparse
import json
import socket
import warnings
from functools import partial
from typing import Any, Callable, Optional

import numpy as np
import tqdm

MID1 = "00200001001000000000"
MID3 = "00200003001000000000"
MID40 = "00200040002000000000"


def connect(
    host: str,
    port: int = 4545,
    timeout: float = 5,
    cur_try: int = 0,
    max_retry: int = 5,
) -> socket.SocketType:
    """
    Connect to the open protocol port in the tool.

    Parameters
    ----------
    host : str
        The IP address of the IxB tool.
    port : int, default: 4545
        The port that open protocol is running at.
    timeout : float, default: 5
        The time in seconds to set the timeout on
        network operations when communicating.
    cur_try : int, default: 0
        The attempt at connecting we are at.
    max_retry : int, default: 5
        The maximum number of attempts to connect we should try.

    Returns
    -------
    sock : socket.SocketType
        A socket object connected to the tool.
    """
    if cur_try >= max_retry:
        raise RecursionError("Maximum attemps to connect exceeded!")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return sock
    except OSError:
        sock.close()
        return connect(host, port, cur_try + 1, max_retry)


def send_mid(message: str, sock: Optional[socket.SocketType] = None, **kwargs):
    """
    Send a message to the tool.
    One attempt at recconecting and sending the message is made.

    Parameters
    ----------
    message : str
        The message to send.
        Should be formatted with the correct header for your MID.
    sock : Optional[socket.SocketType], default: None
        Socket to communicate with the tool.
        If `None` a new connection is made.
    **kwargs
        Arguments to pass to `connect` in case we need to recconect.
    """
    if sock is None:
        sock = connect(**kwargs)
    to_send = (message + chr(0)).encode()
    try:
        sock.sendall(to_send)
    except BrokenPipeError:
        connect(**kwargs)
        sock.sendall(to_send)


def recv_mid(sock: socket.SocketType) -> tuple[str, str, str, bool, bool]:
    """
    Receive a message from the tool.

    Parameters
    ----------
    sock : socket.SocketType
        Socket to communicate with the tool.

    Returns
    -------
    mid : str
        The MID of the returned message.
    rev : str
        The revision of the returned message's MID.
    data : str
        The data in the returned message.
    disconnect : bool
        True if a disconnect event was noticed.
    timeout : bool
        True if a timeout event was noticed.
    """
    keep_receiving = True
    message = ""
    timeout = False
    disconnect = False
    while keep_receiving:
        chunk = ""
        try:
            chunk = sock.recv(1024).decode()
        except socket.timeout:
            timeout = True
            break
        if chunk == "":
            disconnect = True
            break
        message += chunk
        if chunk[-1:] == chr(0):
            keep_receiving = False
    if disconnect:
        warnings.warn("Noticed a disconnect event, you may want to recconect!")
    if timeout:
        warnings.warn("Noticed a timeout event, you may want to recconect!")
    if len(message) < 12:
        raise ValueError("Incomplete header! You may need to recconect!")
    return message[4:8], message[8:11], message[22:-1], disconnect, timeout


def decode0004(message: str) -> str:
    """
    Decode an error message from MID 0004.

    Parameters
    ----------
    message : str
        The message to decode.

    Returns
    -------
    rep : str
        A printable string describing the error.
    """
    mid = message[:4]
    err = message[4:]
    rep = f"MID {mid} failed with error {err}"
    return rep


def decode0002(mid, message):
    """
    Decode a message from MID 0002.

    Parameters
    ----------
    mid : str
        The MID string.
        This should be '0002' or '0004'.
    message : str
        The data returned by `recv` after
        sending MID1.

    Returns
    -------
    rep : str
        A printable string describing the message.
    """
    if mid == "0004":
        return decode0004(message)
    elif mid != "0002":
        raise ValueError("Expected MID 0002 or 0004 but got MID " + mid)
    rep = f"Connected to tool {message[10:35]}"
    return rep


def decode0041(mid, message):
    """
    Decode a message from MID 0041.

    Parameters
    ----------
    mid : str
        The MID string.
        This should be '0041' or '0004'.
    message : str
        The data returned by `recv` after
        sending MID40.

    Returns
    -------
    rep : str
        A printable string describing the message.
    """
    if mid == "0004":
        return decode0004(message)
    elif mid != "0041":
        raise ValueError("Expected MID 0041 or 0004 but got MID " + mid)
    serial = message[:14]
    tightenings = message[16:26]
    cal_date = message[28:38]
    cont_serial = message[49:59]
    cal = float(message[62:67]) / 100
    service = message[69:79]
    tightenings_since = message[91:100]
    firmware = message[115:134]

    rep = f"Tool info:\n\tSerial: {serial}\n\tTotal Tightenings: {tightenings}\n\tLast Calibration: {cal_date}\n\tController Serial: {cont_serial}\n\tCalibration: {cal}\n\tLast Service: {service}\n\tTightenings Since Service: {tightenings_since}\n\tFirmware: {firmware}"
    return rep


def decode2501(mid, message) -> tuple[str, dict[str, Any]]:
    """
    Decode a message from MID 2501.

    Parameters
    ----------
    mid : str
        The MID string.
        This should be '2501' or '0004'.
    message : str
        The data returned by `recv` after
        sending MID2501 via MID6.

    Returns
    -------
    info : str
        If MID 2501 was received then this is the non json info returned.
        If MID 0004 was received then this is a printable error message.
    prog : dict[str, Any]
        A dict generated from the json of the program.
        If MID 0004 was received then an empty dict is returned.
    """
    if mid == "0004":
        return decode0004(message), {}
    elif mid != "2501":
        raise ValueError("Expected MID 2501 or 0004 but got MID " + mid)
    idx = message.find("{")
    if idx == -1:
        raise ValueError("JSON not found in MID 2501 response!")
    info = message[:idx]
    info = info[:-6]
    prog_txt = message[idx:]
    prog = json.loads(prog_txt)

    return info, prog


def construct_2501(pset: int) -> str:
    """
    Construct a message to request MID 2501 via MID 0006.

    Parameters
    ----------
    pset : int
        The id of the program you want.

    Returns
    -------
    message : str
        The message to send to the tool.
    """
    mid = f"003400060010    00  "
    mid += f"250100207{pset:04}2"
    return mid


def construct_2500(pset: int, prog: dict[str, Any]) -> str:
    """
    Construct a message to update a program via MID 2500.

    Parameters
    ----------
    pset : int
        The id of the program to update.
    prog : dict[str, Any]
        The program in dict form.
        This will be dumped to a yaml and sent.

    Returns
    -------
    message : str
        The message to send to the tool.
    """
    prog_txt = json.dumps(prog, separators=(",", ":"))
    info = f"20100101000004010000000{pset:04}{len(prog_txt):06}"
    mid = "00002500002000000000"
    mid += f"{info}{prog_txt}"
    mid = f"{len(mid):04}" + mid[4:]
    return mid


def init(host: str, port: int, **kwargs) -> tuple[
    socket.SocketType,
    Callable[[str], None],
    Callable[[], tuple[str, str, str, bool, bool]],
]:
    """
    Connect to the tool and print identifying information.
    Also generates convenience functions.

    Parameters
    ----------
    host : str
        The IP address of the IxB tool.
    port : int, default: 4545
        The port that open protocol is running at.
    **kwargs
        Additional arguments to pass to `connect`.

    Returns:
    sock : socket.SocketType
        A socket object connected to the tool.
    send : Callable[[str], None]
        A function to send a message to the tool with
        the connection information prepopulated.
    recd : Callable[[], tuple[str, str, str, bool, bool]]
        A function to receive a message from the tool with
        the connection information prepopulated.
    """
    sock = connect(host, port, **kwargs)
    send = partial(send_mid, sock=sock, host=host, port=port)
    recv = partial(recv_mid, sock=sock)
    send(MID1)
    mid, _, dat, _, _ = recv()
    print(decode0002(mid, dat))
    send(MID40)
    mid, _, dat, _, _ = recv()
    print(decode0041(mid, dat))
    return sock, send, recv


def close(sock: socket.SocketType, send: Callable[[str], None]):
    """
    Disconnect from the tool.

    Parameters
    ----------
    sock : socket.SocketType
        A socket object connected to the tool.
    send : Callable[[str], None]
        A function to send a message to the tool with
        the connection information prepopulated.
    """
    send(MID3)
    sock.close()


def get_adjs_names() -> tuple[list[str], list[str], list[str]]:
    """
    Get the names of the adjusters in the order and format of the programs
    on the IxB tool.

    Returns
    -------
    program_names : list[str]
        The names of the programs in the order they will appear on the tool.
    part1 : list[str]
        The names of the adjusters in rows 1-4 of the mirror.
    part2 : list[str]
        The names of the adjusters in rows 5-9 of the mirror.
    """
    adjs = []
    for r in range(1, 10):
        for c in range(1, 10):
            for a in range(1, 6):
                adjs += [f"P{r}{c}V{a}"]

    # There is a dumb 250 limit
    # So we split the mirror into two parts
    # Part 1 is rows 1-4 and part 2 5-9
    split = 4*9*5 # 4 rows * 9 cols * 5 adjusters
    part1 = adjs[:split].copy()
    part2 = adjs[split:].copy()
    adjs = [f"{p1}_{p2}" for p1, p2 in zip(part1, part2)] + part2[split:]

    return adjs, part1, part2


def main():
    # load information
    parser = argparse.ArgumentParser()
    parser.add_argument("adjustments", help="path to adjustments file")
    parser.add_argument(
        "--host", "-H", default="169.254.1.1", type=str, help="the IP of the IxB tool"
    )
    parser.add_argument(
        "--port", "-P", default=4545, type=int, help="the port open protocol runs at"
    )
    parser.add_argument(
        "--microns_per_turn",
        "-m",
        default=100,
        type=float,
        help="The number of microns per turn of the adjuster",
    )
    parser.add_argument(
        "--thresh",
        "-t",
        default=5,
        type=float,
        help="The threshold in microns at which we want to just set the adjustnent to 0",
    )
    args = parser.parse_args()

    # Load the file and build adjustments
    adj_data = np.loadtxt(args.adjustments)
    adjustments = {}
    to_deg = 1000 * 360 / args.microns_per_turn
    thresh = args.thresh * to_deg / 1000
    for adj in adj_data:
        name_root = f"P{int(adj[1])}{int(adj[2])}V"
        p_adj = {f"{name_root}{v}": adj[4 + v] * to_deg for v in range(1, 6)}
        adjustments.update(p_adj)

    # Get the part we want to adjust
    if args.part not in [1, 2]:
        raise ValueError("Part must be 1 or 2")
    adjs = get_adjs_names()[args.part]

    # Connect to tool and send info
    sock, send, recv = init(args.host, args.port)
    sign = {1:1, -1:2}
    failed = []
    to_hit = []
    for i, adj in enumerate(tqdm.tqdm(adjs)):
        send(construct_2501(i + 1))
        mid, _, dat, d, t = recv()
        if mid == "0004" or d or t:
            failed += [adjs]
            sock, send, recv = init(args.host, args.port)
            continue
        _, prog = decode2501(mid, dat)
        ang = adjustments.get(adj, 0.1)
        if np.abs(ang) < thresh:
            ang = .1
        else:
            to_hit += [adj]
        prog["threadDirection"] = sign[np.sign(ang)]
        prog["steps"][1]["stepTightenToAngle"]["angleTarget"] = np.abs(ang)
        send(construct_2500(i + 1, prog))
        mid, _, dat, d, t = recv()
        if mid == "0004" or d or t:
            failed += [adjs]
            sock, send, recv = init(args.host, args.port)
    print(f"failed: {failed}")
    print(f"to_hit: {to_hit}")
    close(sock, send)
