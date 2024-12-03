#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelSinh {
public:
    __aicore__ inline KernelSinh() {}
    __aicore__ inline void Init(/* 开发者填充参数列表 */)
    {
        //考生补充初始化代码
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;

        ASSERT(tileNum != 0 && "tile num can not be zero!");
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE));

    }
    __aicore__ inline void Process()
    {
        //考生补充对“loopCount”的定义，注意对Tiling的处理
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        //考生补充算子代码
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.AllocTensor<DTYPE>();
        AscendC::LocalTensor<DTYPE> yLocal = inQueueY.AllocTensor<DTYPE>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        //考生补充算子计算代码
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.DeQue<DTYPE>();
        AscendC::LocalTensor<DTYPE> yLocal = inQueueY.DeQue<DTYPE>();
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.AllocTensor<DTYPE>();
        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<DTYPE>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);

    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        //考生补充算子代码
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.DeQue<DTYPE>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    //create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;

    //考生补充自定义成员变量
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;

};

extern "C" __global__ __aicore__ void sinh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSinh op;
    //补充init和process函数调用内容
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
