#include <string>
#include <iostream>
#include <torch/script.h>
//#include <torch/torch.h> // maybe not needed?

template<typename T>
class AIModel
{
    private:
        std::string modelFile_; 
        T model_;

    public:
        AIModel() = delete;
        AIModel(std::string modelFile = "")
        {
            initModel(modelFile);
        }

        void initModel(std::string modelFile)
        {
            modelFile_ = modelFile;
            try
            {
                model_ = torch::jit::load(modelFile);
            }
            catch (const c10::Error& e)
            {
                std::cerr << "Error in AIModel while loading torch model from file: " << modelFile << std::endl << e.msg() << std::endl;
            }
        }

        T& getModel()
        {
            return model_;
        }
};
