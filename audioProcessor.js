class NoiseFilterProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.threshold = 0.02;  // 噪声阈值
        // this.port.onmessage = (event) => {
        //     console.log("接收到来自主线程的消息:", event.data);
        // };
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        const output = outputs[0];
        //console.log("处理的音频输出数据:", output[0]);

        // 遍历音频数据，检查音量（RMS）
        for (let channel = 0; channel < input.length; channel++) {
            let inputChannel = input[channel];
            let outputChannel = output[channel];
            let total = 0;

            // 计算均方根（RMS）
            for (let i = 0; i < inputChannel.length; i++) {
                total += inputChannel[i] * inputChannel[i];
            }
            let rms = Math.sqrt(total / inputChannel.length);

            // 如果音量低于阈值，认为是噪音，丢弃音频
            if (rms < this.threshold) {
                // 把输出设为 0，表示丢弃
                outputChannel.fill(0);
                //console.log("检测到噪音，丢弃音频数据");
            } else {
                // 否则将音频数据传递给输出
                outputChannel.set(inputChannel);
                console.log("音频数据通过");

                // 将音频数据转换为 ArrayBuffer 进行传输
                const buffer = new ArrayBuffer(inputChannel.length * 4);  // 每个 float32 占 4 字节
                const view = new Float32Array(buffer);
                view.set(inputChannel);

                // 确保只在有效数据时发送数据
                this.port.postMessage(buffer);  // 发送有效音频数据
                console.log("音频数据:", view);
            }
        }

        return true;  // 返回 true 表示继续处理音频流
    }

    // 检查音频数据是否全为零
    isAudioDataNonZero(data) {
        // 简单检查：如果数据中有非零值，则返回 true
        return data.some(value => value !== 0);
    }
}

// 注册 AudioWorkletProcessor
registerProcessor('noise-filter-processor', NoiseFilterProcessor);
