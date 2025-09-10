# Copyright (c) 2019-2023 Huazhong University of Science and Technology
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation;
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Author: Pengyu Liu <eic_lpy@hust.edu.cn>
#         Hao Yin <haoyin@uw.edu>
#         Muyuan Shen <muyuan_shen@hust.edu.cn>

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)


SIMULATION_EARLY_ENDING = 0.5   # wait and see if the subprocess is running after creation


def get_setting(setting_map: dict[str, Any]) -> str:
    ret = ""
    for key, value in setting_map.items():
        ret += f" --{key}"
        if value is not None:
            ret += f"={value}"
    return ret


def run_single_ns3(
    path,
    pname: str | Path,
    setting: dict[str, Any] | None = None,
    env=None,
    show_output=False,
    debug=False,
):
    if env is None:
        env = {}
    env.update(os.environ)
    env["LD_LIBRARY_PATH"] = os.path.abspath(os.path.join(path, "build", "lib"))
    if Path(pname).is_file():
        cmd = pname
    else:
        exec_path = os.path.join(path, "ns3")
        cmd = f"{exec_path} run {pname} --"
    if setting:
        cmd += get_setting(setting)
    if debug:
        cmd = f"sleep infinity && {cmd}"
    if show_output:
        proc = subprocess.Popen(cmd, shell=True, text=True, env=env, stdin=subprocess.PIPE, preexec_fn=os.setpgrp)
    else:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            text=True,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setpgrp,
        )

    return cmd, proc


# used to kill the ns-3 script process and its child processes
def kill_proc_tree(p: subprocess.Popen, timeout: int = 1):
    logger.info('ns3ai_utils: Killing subprocesses...')
    p = psutil.Process(p.pid)
    ch = p.children(recursive=True) + [p]
    for c in ch:
        try:
            # print("\t-- {}, pid={}, ppid={}".format(psutil.Process(c.pid).name(), c.pid, c.ppid()))
            # print("\t   \"{}\"".format(" ".join(c.cmdline())))
            c.kill()
            c.wait(timeout=timeout)
        except Exception:
            continue
    

# According to Python signal docs, after a signal is received, the
# low-level signal handler sets a flag which tells the virtual machine
# to execute the corresponding Python signal handler at a later point.
#
# As a result, a long ns-3 simulation, during which no C++-Python
# interaction occurs (such as the Multi-BSS example), may run uninterrupted
# for a long time regardless of any signals received.
def sigint_handler(sig, frame):
    print("\nns3ai_utils: SIGINT detected")
    exit(1)  # this will execute the `finally` block


# This class sets up the shared memory and runs the simulation process.
class Experiment:
    _created = False

    # init ns-3 environment
    # \param[in] memSize : share memory size
    # \param[in] targetName : program name of ns3
    # \param[in] path : current working directory
    def __init__(
        self,
        targetName: str | Path,
        ns3Path,
        msgModule,
        debug=False,
        handleFinish=False,
        useVector=False,
        vectorSize=None,
        shmSize=4096,
        segName: str = "ns3-ai",
    ):
        if self._created:
            raise Exception('ns3ai_utils: Error: Experiment is singleton')
        self._created = True
        self.targetName = targetName  # ns-3 target name or file name
        self.debug = debug
        os.chdir(ns3Path)
        self.msgModule = msgModule
        self.handleFinish = handleFinish
        self.useVector = useVector
        self.vectorSize = vectorSize
        self.shmSize = shmSize
        self.segName = segName

        self.msgInterface = msgModule.Ns3AiMsgInterfaceImpl(
            True,
            self.useVector,
            self.handleFinish,
            self.shmSize,
            self.segName,
            segName + "2py",
            segName + "2cpp",
            segName + "lock",
        )
        if self.useVector:
            if self.vectorSize is None:
                raise Exception('ns3ai_utils: Error: Using vector but size is unknown')
            self.msgInterface.GetCpp2PyVector().resize(self.vectorSize)
            self.msgInterface.GetPy2CppVector().resize(self.vectorSize)

        self.proc = None
        self.simCmd = None
        logger.info('ns3ai_utils: Experiment initialized')

    def __del__(self):
        self.kill()
        del self.msgInterface
        logger.info('ns3ai_utils: Experiment destroyed')

    # run ns3 script in cmd with the setting being input
    # \param[in] setting : ns3 script input parameters(default : None)
    # \param[in] show_output : whether to show output or not(default : False)
    def run(self, setting: dict[str, Any] = None, show_output=False):
        self.kill()
        self.simCmd, self.proc = run_single_ns3(
            "./",
            self.targetName,
            setting=setting,
            show_output=show_output,
            debug=self.debug,
        )
        logger.info("ns3ai_utils: Running ns-3 with: %s", self.simCmd)
        # exit if an early error occurred, such as wrong target name
        time.sleep(SIMULATION_EARLY_ENDING)
        if not self.isalive():
            logger.info('ns3ai_utils: Subprocess died very early')
            exit(1)
        return self.msgInterface

    def kill(self):
        if self.proc and self.isalive():
            kill_proc_tree(self.proc)
            self.proc = None
            self.simCmd = None

    def isalive(self):
        return self.proc.poll() is None


__all__ = ['Experiment']
