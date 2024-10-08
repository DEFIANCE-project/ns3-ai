from contextlib import suppress
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Any

import gymnasium as gym
import messages_pb2 as pb
import ns3ai_gym_msg_py as py_binding
import numpy as np
from gymnasium import spaces
from ns3ai_utils import Experiment


class Ns3Env(gym.Env):
    _created = False

    def _create_space(self, spaceDesc):
        space = None
        if spaceDesc.type == pb.Discrete:
            discreteSpacePb = pb.DiscreteSpace()
            spaceDesc.space.Unpack(discreteSpacePb)
            space = spaces.Discrete(discreteSpacePb.n)

        elif spaceDesc.type == pb.Box:
            boxSpacePb = pb.BoxSpace()
            spaceDesc.space.Unpack(boxSpacePb)
            low = np.array(boxSpacePb.lows) if boxSpacePb.lows else boxSpacePb.low
            high = np.array(boxSpacePb.highs) if boxSpacePb.highs else boxSpacePb.high
            shape = tuple(boxSpacePb.shape)
            mtype = boxSpacePb.dtype

            if mtype == pb.INT:
                mtype = int
            elif mtype == pb.UINT:
                raise NotImplementedError(
                    "uint is not supported by all rl frameworks. Use int instead!"
                )
            elif mtype == pb.DOUBLE:
                mtype = np.float64
            else:
                mtype = np.float32

            space = spaces.Box(low=low, high=high, shape=shape, dtype=mtype)

        elif spaceDesc.type == pb.Tuple:
            mySpaceList = []
            tupleSpacePb = pb.TupleSpace()
            spaceDesc.space.Unpack(tupleSpacePb)

            for pbSubSpaceDesc in tupleSpacePb.element:
                subSpace = self._create_space(pbSubSpaceDesc)
                mySpaceList.append(subSpace)

            mySpaceTuple = tuple(mySpaceList)
            space = spaces.Tuple(mySpaceTuple)

        elif spaceDesc.type == pb.Dict:
            mySpaceDict = {}
            dictSpacePb = pb.DictSpace()
            spaceDesc.space.Unpack(dictSpacePb)

            for pbSubSpaceDesc in dictSpacePb.element:
                subSpace = self._create_space(pbSubSpaceDesc)
                mySpaceDict[pbSubSpaceDesc.name] = subSpace

            space = spaces.Dict(mySpaceDict)

        return space

    def _create_data(self, dataContainerPb):
        if dataContainerPb.type == pb.Discrete:
            discreteContainerPb = pb.DiscreteDataContainer()
            dataContainerPb.data.Unpack(discreteContainerPb)
            data = discreteContainerPb.data
            return data

        if dataContainerPb.type == pb.Box:
            boxContainerPb = pb.BoxDataContainer()
            dataContainerPb.data.Unpack(boxContainerPb)
            # print(boxContainerPb.shape, boxContainerPb.dtype, boxContainerPb.uintData)

            if boxContainerPb.dtype == pb.INT:
                data = np.array(boxContainerPb.intData, dtype=int)
            elif boxContainerPb.dtype == pb.UINT:
                data = np.array(boxContainerPb.uintData, dtype=np.uint)
            elif boxContainerPb.dtype == pb.DOUBLE:
                data = np.array(boxContainerPb.doubleData, dtype=np.float64)
            else:
                data = np.array(boxContainerPb.floatData, dtype=np.float32)

            return data.reshape(boxContainerPb.shape)

        elif dataContainerPb.type == pb.Tuple:
            tupleDataPb = pb.TupleDataContainer()
            dataContainerPb.data.Unpack(tupleDataPb)

            myDataList = []
            for pbSubData in tupleDataPb.element:
                subData = self._create_data(pbSubData)
                myDataList.append(subData)

            data = tuple(myDataList)
            return data

        elif dataContainerPb.type == pb.Dict:
            dictDataPb = pb.DictDataContainer()
            dataContainerPb.data.Unpack(dictDataPb)

            myDataDict = {}
            for pbSubData in dictDataPb.element:
                subData = self._create_data(pbSubData)
                myDataDict[pbSubData.name] = subData

            data = myDataDict
            return data

    def initialize_env(self):
        simInitMsg = pb.SimInitMsg()
        self.msgInterface.PyRecvBegin()
        request = self.msgInterface.GetCpp2PyStruct().get_buffer()
        simInitMsg.ParseFromString(request)
        self.msgInterface.PyRecvEnd()

        self.action_space = self._create_space(simInitMsg.actSpace)
        self.observation_space = self._create_space(simInitMsg.obsSpace)

        reply = pb.SimInitAck()
        reply.done = True
        reply.stopSimReq = False
        reply_str = reply.SerializeToString()
        assert len(reply_str) <= py_binding.msg_buffer_size

        self.msgInterface.PySendBegin()
        self.msgInterface.GetPy2CppStruct().size = len(reply_str)
        self.msgInterface.GetPy2CppStruct().get_buffer_full()[: len(reply_str)] = (
            reply_str
        )
        self.msgInterface.PySendEnd()
        return True

    def send_close_command(self):
        reply = pb.EnvActMsg()
        reply.stopSimReq = True

        replyMsg = reply.SerializeToString()
        assert len(replyMsg) <= py_binding.msg_buffer_size
        self.msgInterface.PySendBegin()
        self.msgInterface.GetPy2CppStruct().size = len(replyMsg)
        self.msgInterface.GetPy2CppStruct().get_buffer_full()[: len(replyMsg)] = (
            replyMsg
        )
        self.msgInterface.PySendEnd()

        self.newStateRx = False
        return True

    def rx_env_state(self):
        if self.newStateRx:
            return

        envStateMsg = pb.EnvStateMsg()
        self.msgInterface.PyRecvBegin()
        request = self.msgInterface.GetCpp2PyStruct().get_buffer()
        envStateMsg.ParseFromString(request)
        self.msgInterface.PyRecvEnd()

        self.obsData = self._create_data(envStateMsg.obsData)
        self.reward = envStateMsg.reward
        self.gameOver = envStateMsg.isGameOver
        self.gameOverReason = envStateMsg.reason

        if self.gameOver:
            self.send_close_command()

        self.extraInfo = dict(envStateMsg.info)

        self.newStateRx = True

    def get_obs(self):
        return self.obsData

    def get_reward(self):
        return self.reward

    def is_game_over(self):
        return self.gameOver

    def get_extra_info(self):
        return self.extraInfo

    def _pack_data(self, actions, spaceDesc):
        dataContainer = pb.DataContainer()

        spaceType = spaceDesc.__class__

        if spaceType == spaces.Discrete:
            dataContainer.type = pb.Discrete
            discreteContainerPb = pb.DiscreteDataContainer()
            discreteContainerPb.data = actions
            dataContainer.data.Pack(discreteContainerPb)

        elif spaceType == spaces.Box:
            dataContainer.type = pb.Box
            boxContainerPb = pb.BoxDataContainer()
            shape = [len(actions)]
            boxContainerPb.shape.extend(shape)

            if spaceDesc.dtype in ["int", "int8", "int16", "int32", "int64"]:
                boxContainerPb.dtype = pb.INT
                boxContainerPb.intData.extend(actions)

            elif spaceDesc.dtype in ["uint", "uint8", "uint16", "uint32", "uint64"]:
                raise NotImplementedError(
                    "uint is not supported by all rl frameworks. Use int instead!"
                )

            elif spaceDesc.dtype.name in ["float", "float32"]:
                boxContainerPb.dtype = pb.FLOAT
                boxContainerPb.floatData.extend(actions)

            elif spaceDesc.dtype.name in ["double", "float64"]:
                boxContainerPb.dtype = pb.DOUBLE
                boxContainerPb.doubleData.extend(actions)

            else:
                boxContainerPb.dtype = pb.FLOAT
                boxContainerPb.floatData.extend(actions)

            dataContainer.data.Pack(boxContainerPb)

        elif spaceType == spaces.Tuple:
            dataContainer.type = pb.Tuple
            tupleDataPb = pb.TupleDataContainer()

            spaceList = list(spaceDesc.spaces)
            subDataList = []
            for subAction, subActSpaceType in zip(actions, spaceList):
                subData = self._pack_data(subAction, subActSpaceType)
                subDataList.append(subData)

            tupleDataPb.element.extend(subDataList)
            dataContainer.data.Pack(tupleDataPb)

        elif spaceType == spaces.Dict:
            dataContainer.type = pb.Dict
            dictDataPb = pb.DictDataContainer()

            subDataList = []
            for sName, subAction in actions.items():
                subActSpaceType = spaceDesc.spaces[sName]
                subData = self._pack_data(subAction, subActSpaceType)
                subData.name = sName
                subDataList.append(subData)

            dictDataPb.element.extend(subDataList)
            dataContainer.data.Pack(dictDataPb)

        return dataContainer

    def send_actions(self, actions):
        reply = pb.EnvActMsg()

        actionMsg = self._pack_data(actions, self.action_space)
        reply.actData.CopyFrom(actionMsg)

        replyMsg = reply.SerializeToString()
        assert len(replyMsg) <= py_binding.msg_buffer_size
        self.msgInterface.PySendBegin()
        self.msgInterface.GetPy2CppStruct().size = len(replyMsg)
        self.msgInterface.GetPy2CppStruct().get_buffer_full()[: len(replyMsg)] = (
            replyMsg
        )
        self.msgInterface.PySendEnd()
        self.newStateRx = False
        return True

    def get_state(self):
        obs = self.get_obs()
        reward = self.get_reward()
        terminated = False
        truncated = False
        if self.is_game_over():
            if self.gameOverReason == 1:
                terminated = True  # end because the agent reached its final state
            else:
                truncated = True  # end because the simulation ended (for this agent)
        extraInfo = self.get_extra_info()
        return obs, reward, terminated, truncated, extraInfo

    def __init__(
        self,
        targetName: str | Path,
        ns3Path: str,
        ns3Settings: dict[str, Any] | None = None,
        debug: bool = False,
        shmSize=8192,
        segName="My Seg",  # the names for the shared memory segments used by boost
        cpp2pyMsgName="My Cpp to Python Msg",
        py2cppMsgName="My Python to Cpp Msg",
        lockableName="My Lockable",
        trial_name: str | None = None,
        simulation_wd: Path | None = None,
        runStep: int = 1,
    ):
        if self._created:
            raise Exception("Error: Ns3Env is singleton")
        self.targetName = targetName
        self.debug = debug
        self.shmSize = shmSize
        self._created = True
        self.ns3Settings = ns3Settings
        self.simulation_wd = simulation_wd
        self.runStep = runStep
        if trial_name is not None:
            # indexing the memory segments with the trial name to allow parallel execution of ns3 environments
            self.ns3Settings["trial_name"] = (
                trial_name  # add trial name to the command line arguments so the ns3 process can use it
            )
            segName = segName + trial_name
            cpp2pyMsgName = cpp2pyMsgName + trial_name
            py2cppMsgName = py2cppMsgName + trial_name
            lockableName = lockableName + trial_name

        self.exp = Experiment(
            targetName,
            ns3Path,
            py_binding,
            debug=debug,
            shmSize=shmSize,
            segName=segName,
            cpp2pyMsgName=cpp2pyMsgName,
            py2cppMsgName=py2cppMsgName,
            lockableName=lockableName,
            simulation_wd=simulation_wd,
        )

        self.newStateRx = False
        self.obsData = None
        self.reward = 0
        self.gameOver = False
        self.gameOverReason = None
        self.extraInfo = None

        self.msgInterface = self.exp.run(setting=self.ns3Settings, show_output=True)
        self.initialize_env()
        # get first observations
        self.rx_env_state()
        self.envDirty = False

    def step(self, actions):
        self.send_actions(actions)
        self.rx_env_state()
        self.envDirty = True
        return self.get_state()

    def reset(self, seed=None, options=None):
        if not self.envDirty:
            obs = self.get_obs()
            return obs, {}

        # not using self.exp.kill() here in order for semaphores to reset to initial state
        if not self.gameOver:
            self.rx_env_state()
            self.send_close_command()
            with suppress(TimeoutExpired):
                self.exp.proc.wait(2)

        self.msgInterface = None
        self.newStateRx = False
        self.obsData = None
        self.reward = 0
        self.gameOver = False
        self.gameOverReason = None
        self.extraInfo = None

        # Allow the user to increment the run number on environment reset. This way the random variables used inside the simulation will use different values. This is required for reproducibility and to avoid overfitting.
        if "runId" in self.ns3Settings:
            self.ns3Settings["runId"] = int(self.ns3Settings["runId"]) + self.runStep

        self.msgInterface = self.exp.run(setting=self.ns3Settings, show_output=True)
        self.initialize_env()
        # get first observations
        self.rx_env_state()
        self.envDirty = False

        obs = self.get_obs()
        return obs, {}

    def render(self, mode="human"):
        return

    def get_random_action(self):
        act = self.action_space.sample()
        return act

    def close(self):
        if not self.gameOver:
            self.rx_env_state()
            self.send_close_command()
            with suppress(TimeoutExpired):
                self.exp.proc.wait(2)

        # environment is not needed anymore, so kill subprocess in a straightforward way
        self.exp.kill()
        # destroy the message interface and its shared memory segment
        del self.exp

    def __getstate__(self):
        return {
            "targetName": self.targetName,
            "ns3Path": ".",
            "ns3Settings": self.ns3Settings,
            "debug": self.debug,
            "shmSize": self.shmSize,
            "simulation_wd": self.simulation_wd,
        }

    def __setstate__(self, state):
        if hasattr(self, "exp"):
            self.close()
        self.__init__(**state)
