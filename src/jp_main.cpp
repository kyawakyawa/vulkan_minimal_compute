#include <vulkan/vulkan.h>

#include <vector>
#include <string.h>
#include <assert.h>
#include <stdexcept>
#include <cmath>

#include "lodepng.h"

constexpr  int32_t kWidth          = 3200;       // マンデルブロ集合のレンダリング結果の幅
constexpr  int32_t kHeight         = 2400;       // マンデルブロ集合のレンダリング結果の高さ
constexpr  int32_t kWorkgroupeSize = 32;         // Workgroup の大きさ (Nvidia の Warp)

#ifdef NDEBUG
constexpr bool kEnableValidationLayers = false;
#else
constexpr bool kEnableValidationLayers = true;
#endif

// Vulkan のAPIが返す引数をもらってエラーチェックを行う
#define VK_CHECK_RESULT(f){                                                        \
  VkResult res = (f);                                                              \
  if (res != VK_SUCCESS) {                                                         \
    printf("Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
    assert(res == VK_SUCCESS);                                                     \
  }                                                                                \
}


/*
 * このプログラムはマンデルブロ集合をレンダリングするcompute shader を起動し、
 * ストレージバッファにレンダリングする。
 * ストレージバッファはGPUから読み込まれ、pngとして保存される
 */
class ComputeApplication {
  private:
    // ピクセルのデータ構造
    struct Pixel {
      float r, g, b, a;
    };

    /*
     * Vulkanを使うためにはインスタンスを作成する必要がある
     */
    VkInstance instance;

    // TODO : add comment
    VkDebugReportCallbackEXT debug_report_callback;

    /*
     * 物理デバイスはVulkanをサポートしているデバイス
     */ 
    VkPhysicalDevice physical_device;

    /*
     * 論理デバイス、これによって物理デバイスとやり取りをすることが出来るようになる
     */
    VkDevice device;

    /*
     * Vulkanですべてのグラフィックコマンドと計算コマンドが通るパイプライン指定する
     * このプログラムではシンプルな計算パイプラインを作る
     */
    VkPipeline pipeline;
    VkPipelineLayout pipeline_layout;
    VkShaderModule compute_shader_module;

    /*
     * コマンドバッファはコマンドを記録する
     * このコマンドはキューからサブミットされる
     * コマンドプールを使ってコマンドバッファを確保する
     */
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    /*
     * Descriptors はシェーダー内のリソースを表す
     * uniform バッファ、storage バッファ image をGLSL内で使うことを可能にする
     */
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;
    VkDescriptorSetLayout descriptor_set_layout;

    /*
     * マンデルブロ集合はこのバッファにレンダリングされる
     * バッファをバックアップするのがバッファメモリである
     */
    VkBuffer buffer;
    VkDeviceMemory buffer_memory;

    // バッファのサイズ(byte)
    uint32_t buffer_size;

    // 有効にしたレイヤーの名前
    std::vector<const char *> enabled_layers;

    /*
     * GPU上でコマンドを実行するために、このキューにコマンドがサブミットされる必要がある
     * コマンドはコマンドバッファに格納される
     * このコマンドはキューから与えられる
     *
     * デバイスに異なる種類のキューがある
     * グラフィックの処理をサポートするだけではない
     * このプログラムではキューは計算処理をサポートしている
     */
    VkQueue queue; // 計算処理をサポートするキュー

    /*
     * 同じ機能を持つキューのグループはキューファミリーとしてまとめられる
     * (グラフィックをサポートするものとか計算処理をサポートするものとか)
     *
     * コマンドバッファにサブミットするときファミリーのどのキューからサブミットするかを指定する必要がある
     * この変数はファミリーでのキューの添字番号が格納される
     */
    uint32_t queue_family_index;

  public:
    void Run() {
      // ストレージバッファの大きさ(レンダリングされたマンデルブロ集合が格納される)
      buffer_size = sizeof(Pixel) * kWidth * kHeight;

      // Vulkan を初期化する
      CreateInstance();
      FindPhysicalDevice();
      CreateDevice();
      CreateBuffer();
      CreateDescriptorSetLayout();
      CreateDescriptorSet();
      CreateComputePipeline();
      CreateCommandBuffer();

      // 最後に記録されたコマンドバッファの中身を実行する
      RunCommandBuffer();

      // The former command rendered a mandelbrot set to a buffer.
      // Save that buffer as a png on disk.
      // レンダリングされたマンデルブロ集合をバッファに入れる
      // バッファはpngとしてディスクに保存される
      SaveRenderedImage();

      // Vulkanの全てのリソースを解放する
      CleanUp();
    }

    void CreateInstance() {
      /*
       * 有効な拡張の名前
       */
      std::vector<const char *> enabled_extensions;

      /*
       * 検証レイヤーを有効にすることで、APIが不正に使われた時に警告が出されるようになる
       * VK_LAYER_LUNARG_standard_validationレイヤーを有効にする
       * これは基本的な便利な検証レイヤーの集まり
       */
      if (kEnableValidationLayers) {
        /*
         * vkEnumerateInstanceLayerProperties()関数で全てのレイヤーを取得する
         */
        uint32_t layer_count;
        vkEnumerateInstanceLayerProperties(&layer_count, nullptr);// nullptrを渡すとレイヤーの数を取得できる

        std::vector<VkLayerProperties> layer_properties(layer_count);
        vkEnumerateInstanceLayerProperties(&layer_count, layer_properties.data());

        /*
         * サポートされるレイヤーにVK_LAYER_LUNARG_standard_validtionが含まれるか調べる
         */
        bool found_layer = false;
        for (VkLayerProperties prop : layer_properties) {

          if (strcmp("VK_LAYER_LUNARG_standard_validation", prop.layerName) == 0) {
            found_layer = true;
          }

        }

        if (!found_layer) {
          throw std::runtime_error("Layer VK_LAYER_LUNARG_standard_validation not supported\n");
        }
        enabled_layers.push_back("VK_LAYER_LUNARG_standard_validation"); // このレイヤーを使うことが出来る

        /*
         * 検証レイヤーが発する警告を出力するために
         * VK_EXT_DEBUG_REPORT_EXTENSION_NAMEという 名前の拡張を有効化する
         *
         * よって、もう一度この拡張がサポートされているかチェックする
         */

        uint32_t extension_count;

        vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
        std::vector<VkExtensionProperties> extension_properties(extension_count);
        vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extension_properties.data());

        bool found_extension = false;
        for (VkExtensionProperties prop : extension_properties) {
          if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, prop.extensionName) == 0) {
            found_extension = true;
            break;
          }

        }

        if (!found_extension) {
          throw std::runtime_error("Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported\n");
        }
        enabled_extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
      }		

      /*
       * 次に実際にインスタンスを作っていく
       */

      /*
       * アプリケーションの情報を格納する
       * これは実際にはそこまでが大事ではない
       * 唯一重要なのは apiVersionである
       */
      VkApplicationInfo application_info = {};
      application_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
      application_info.pApplicationName   = "Hello world app";
      application_info.applicationVersion = 0;
      application_info.pEngineName        = "awesomeengine";
      application_info.engineVersion      = 0;
      application_info.apiVersion         = VK_API_VERSION_1_0;;

      /*
       * インスタンスを作る時の情報を入れる
       */
      VkInstanceCreateInfo create_info = {};
      create_info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
      create_info.flags                   = 0;
      create_info.pApplicationInfo        = &application_info;

      // vulkan に使いたいレイヤーと拡張を渡す
      create_info.enabledLayerCount       = enabled_layers.size();
      create_info.ppEnabledLayerNames     = enabled_layers.data();
      create_info.enabledExtensionCount   = enabled_extensions.size();
      create_info.ppEnabledExtensionNames = enabled_extensions.data();

      /*
       * 実際にインスタンスを作る
       * インスタンスを作ると、vulkanを使い始めることが出来る
       */
      VK_CHECK_RESULT(vkCreateInstance(
            &create_info,
            nullptr,
            &instance));

      /*
       * VK_EXT_DEBUG_REPORT_EXTENSION_NAMEで使うコールバック関数を登録する
       * これによって検証レイヤーが発する警告を出力できる
       */
      if (kEnableValidationLayers) {
        VkDebugReportCallbackCreateInfoEXT create_info = {};
        create_info.sType       = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
        create_info.flags       = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
        create_info.pfnCallback = &DebugReportCallbackFn;//コールバック関数

        // 明示的にこの関数を読み込む必要がある
        auto vk_create_debug_report_callback_ext = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
        if (vk_create_debug_report_callback_ext == nullptr) {
          throw std::runtime_error("Could not load vkCreateDebugReportCallbackEXT");
        }

        // コールバック関数を作成し登録する
        VK_CHECK_RESULT(vk_create_debug_report_callback_ext(instance, &create_info, nullptr, &debug_report_callback));
      }

    }

    void FindPhysicalDevice() {
      /*
       * この関数でVulkanで使える物理デバイスを調べる
       */

      /*
       * まず、vkEnumeratePhysicalDevices関数を用いて物理デバイスをリスト化する
       */
      uint32_t device_count;
      vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
      if (device_count == 0) {
        throw std::runtime_error("could not find a device with vulkan support");
      }

      std::vector<VkPhysicalDevice> devices(device_count);
      vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

      /*
       * 次に目的のデバイスを選ぶ
       *
       * VkPhysicalDeviceFeatures()関数で、デバイスでサポートされている物理機能の詳細なリストを取得できる
       * このデモでは単にcompute shader しか起動しないので、特別な物理的な要求はない
       * VkPhysicalDeviceProperties()関数で、物理デバイスのプロパティのリストを得る
       * 一番大事なことは物理デバイスの制約を入手することである
       * このプログラムではcompute shaderを起動するのでワークグループの最大の大きさや
       * compute shader の最大呼び出し数が物理デバイスによって制限される
       * maxComputeWorkGroupCount,maxComputeWorkGroupInvocations,maxComputeWorkGroupSizeを超えていないか確認する必要がある
       * 更にcompute shaderで用いるストレージバッファが大きくなりすぎないように maxStorageBufferRange で制約を確認する必要がある
       * しかし、このプログラムでこれらはそこまで大きくなく、ほとんどのデバイスで扱える
       *
       * また簡単にするためにチェックは行わず、リストの最初のデバイスを選択するとする
       * 本格的なプログラムではこれらを考える必要がある
       */
      for (VkPhysicalDevice device : devices) {
        if (true) { // ここではチェックしないのでtrueになる
          physical_device = device;
          break;
        }
      }
    }

    void CreateDevice() {
      /*
       * この関数で論理デバイスを作る
       */

      /*
       *  デバイスを作る時、どのキューを持つか指定する
       */
      VkDeviceQueueCreateInfo queue_create_info = {};
      queue_create_info.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queue_family_index                 = GetComputeQueueFamilyIndex();                    // 計算処理用のキューファミリーを見つける
      queue_create_info.queueFamilyIndex = queue_family_index;
      queue_create_info.queueCount       = 1;                                     // キューファミリーから一つキューを作る これ以上はいらない
      float queue_priorities             = 1.0;                                         // 一つのキューしか使わないのでこれは大事ではない
      queue_create_info.pQueuePriorities = &queue_priorities;
      /*
       * 論理デバイスを作る
       * 論理デバイスは物理デバイスと対話することを可能にする
       */
      VkDeviceCreateInfo device_create_info = {};

      // 望まれるデバイスの特性を指定する このプログラムでは必要でない
      VkPhysicalDeviceFeatures device_features = {};

      device_create_info.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
      device_create_info.enabledLayerCount    = enabled_layers.size();  // ここで検証レイヤーを指定する必要がある
      device_create_info.ppEnabledLayerNames  = enabled_layers.data();
      device_create_info.pQueueCreateInfos    = &queue_create_info; // 論理デバイスを作る時キューも指定する必要がある
      device_create_info.queueCreateInfoCount = 1;
      device_create_info.pEnabledFeatures     = &device_features;

      VK_CHECK_RESULT(vkCreateDevice(physical_device, &device_create_info, nullptr, &device)); // 論理デバイスを作成

      // Get a handle to the only member of the queue family.
      // TODO (kyawakyawa) : add japanese comment
      vkGetDeviceQueue(device, queue_family_index, 0, &queue);
    }

    void CreateBuffer() {
      /*
       * バッファを作成する
       * compute shader 内でこのバッファにマンデルブロ集合をレンダリングする
       */

      VkBufferCreateInfo buffer_create_info = {};
      buffer_create_info.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      buffer_create_info.size               = buffer_size; // バッファのサイズ(byte)
      buffer_create_info.usage              = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // バッファはストレージバッファとして用いられる
      buffer_create_info.sharingMode        = VK_SHARING_MODE_EXCLUSIVE; // バッファは一度に一つのキューファミリーに独占される

      VK_CHECK_RESULT(vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer)); // バッファを作成する

      /*
       * バッファ自身でメモリを確保しないので、手動で確保する必要がある
       */

      /*
       * まずバッファが要求するメモリ要件を調べる
       */
      VkMemoryRequirements memory_requirements;
      vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);

      /*
       * バッファのためメモリを確保のためにメモリ要件を用いる
       */
      VkMemoryAllocateInfo allocate_info = {};
      allocate_info.sType                = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocate_info.allocationSize       = memory_requirements.size; // 要求されたメモリを指定する
      /*
       * 確保できるメモリにはいくつか種類があり、選択する必要がある
       *
       * 1) メモリ要件を満たすもの(memory_requirements.memoryTypeBit)
       * 2) このプログラムの用途を満たすもの
       *
       * (vkMapMemoryを使ってGPUからCPUにバッファメモリを読み込めるようするため、
       *     VK_MEMORY_PROPERTY_HOST_VISIBLE_BITを設定する)
       *
       * また、VK_MEMORY_PROPERTY_HOST_COHERENT_BITを設定しておくと、
       * デバイス（GPU）によって書き込まれたメモリは、余分なフラッシュコマンドを呼び出さなくても、
       * ホスト（CPU）から簡単に見えるようになる
       * 従って、便利なので、このフラグを設定する。
       */
      allocate_info.memoryTypeIndex = FindMemoryType(
          memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

      VK_CHECK_RESULT(vkAllocateMemory(device, &allocate_info, nullptr, &buffer_memory)); // デバイス上のメモリを確保する

      /* 
       * 確保したメモリとバッファを関連付ける
       * これによって実際のメモリによってバッファが使えるようになる
       */
      VK_CHECK_RESULT(vkBindBufferMemory(device, buffer, buffer_memory, 0));
    }

    void CreateDescriptorSetLayout() {
      /*
       * この関数で  descriptorの集まりのレイアウトを指定する
       * これは descriptorとシェーダーのリソースを結びつけることを可能にする
       */

      /*
       * 結合点0に結びつける型をしてする(今回は VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
       * これはcompute shader の
       *
       * layout(std140, binding = 0) buffer buf
       * 
       * に結び付けられる
       */

      VkDescriptorSetLayoutBinding descriptor_set_layout_binding = {};
      descriptor_set_layout_binding.binding         = 0; // binding = 0
      descriptor_set_layout_binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_set_layout_binding.descriptorCount = 1;
      descriptor_set_layout_binding.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

      VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = {};
      descriptor_set_layout_create_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      descriptor_set_layout_create_info.bindingCount = 1; // 紐づける VkDescriptorSetLayoutBinding は一つののみ
      descriptor_set_layout_create_info.pBindings    = &descriptor_set_layout_binding; 

      // Descriptorの集まりのレイアウトを作成する
      VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptor_set_layout_create_info, NULL, &descriptor_set_layout));
    }

    void CreateDescriptorSet() {
      /*
       * この関数でdescriptorの集まりを確保する
       * まずdescriptorプールを作る
       */

      /*
       * このdescriptorプールは一つのストレージバッファのみ確保出来る
       */
      VkDescriptorPoolSize descriptor_pool_size = {};
      descriptor_pool_size.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_pool_size.descriptorCount = 1;

      VkDescriptorPoolCreateInfo descriptor_pool_create_info = {};
      descriptor_pool_create_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      descriptor_pool_create_info.maxSets       = 1; // プールから一つのdescriptorの集まりのみ確保する必要がある
      descriptor_pool_create_info.poolSizeCount = 1;
      descriptor_pool_create_info.pPoolSizes    = &descriptor_pool_size;

      // descriptorを作成する
      VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptor_pool_create_info, nullptr, &descriptor_pool));

      /*
       * プールが確保されたら、descriptorの集まりを確保する
       */
      VkDescriptorSetAllocateInfo descriptor_set_allocate_info = {};
      descriptor_set_allocate_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; 
      descriptor_set_allocate_info.descriptorPool     = descriptor_pool; // どのプールから確保するか
      descriptor_set_allocate_info.descriptorSetCount = 1;               // 一つのdescriptorの集まりを確保する
      descriptor_set_allocate_info.pSetLayouts        = &descriptor_set_layout;

      // allocate descriptor set.
      //  descriptorの集まりを確保する
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptor_set_allocate_info, &descriptor_set));

      /*
       * descriptorとストレージバッファをつなげる
       * descriptorの集まりを更新するために vkUpdateDescriptorSets()関数を用いる
       */

      // Specify the buffer to bind to the descriptor.
      // descriptorに結びつけるバッファを指定する
      VkDescriptorBufferInfo descriptor_buffer_info = {};
      descriptor_buffer_info.buffer = buffer;
      descriptor_buffer_info.offset = 0;
      descriptor_buffer_info.range  = buffer_size;

      VkWriteDescriptorSet write_descriptor_set = {};
      write_descriptor_set.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      write_descriptor_set.dstSet          = descriptor_set; // 書き込むdescriptorの集まり
      write_descriptor_set.dstBinding      = 0; // 最初のものに書き込み、紐付ける
      write_descriptor_set.descriptorCount = 1; // 一つのdescriptorを更新する
      write_descriptor_set.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // ストレージバッファ
      write_descriptor_set.pBufferInfo     = &descriptor_buffer_info;

      // descriptorの集合の更新を行う
      vkUpdateDescriptorSets(device, 1, &write_descriptor_set, 0, nullptr);
    }

    void CreateComputePipeline() {
      /*
       * この関数でcompute pipeline を作る
       */

      /*
       * shaderモデルを作成する
       * shaderモデルは基本的にはshaderのコード(GLSL)でカプセル化される
       */
      uint32_t file_length;

      /* 
       * GLSLやHLSLからSPIR-Vを生成するにはglslangValidatorを用いる
       */

      uint32_t* code = ReadFile(file_length, "shaders/comp.spv");

      VkShaderModuleCreateInfo create_info = {};
      create_info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      create_info.pCode    = code;
      create_info.codeSize = file_length;

      VK_CHECK_RESULT(vkCreateShaderModule(device, &create_info, nullptr, &compute_shader_module));
      delete[] code;

      /*
       * computeパイプラインを作る
       * graphicsパイプラインよりcomputeパイプラインはシンプルである
       * compute shaderは一つのステージのみである
       *
       * まずcomputeシェーダーのステージを指定する
       * エントリーポイントは main
       */
      VkPipelineShaderStageCreateInfo shader_stage_create_info = {};
      shader_stage_create_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      shader_stage_create_info.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
      shader_stage_create_info.module = compute_shader_module;
      shader_stage_create_info.pName  = "main";

      /*
       * パイプラインレイアウトはパイプラインがdescriptor の集まりにアクセスすることを可能にする
       * よって先に作ったdescriptor の集まりのレイアウトを指定する
       */
      VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
      pipeline_layout_create_info.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pipeline_layout_create_info.setLayoutCount = 1;
      pipeline_layout_create_info.pSetLayouts    = &descriptor_set_layout; 
      VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipeline_layout_create_info, nullptr, &pipeline_layout));

      VkComputePipelineCreateInfo pipeline_create_info = {};
      pipeline_create_info.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      pipeline_create_info.stage  = shader_stage_create_info;
      pipeline_create_info.layout = pipeline_layout;

      /*
       * 最後にcomputeパイプラインを作成する
       */
      VK_CHECK_RESULT(vkCreateComputePipelines(
            device, VK_NULL_HANDLE,
            1, &pipeline_create_info,
            NULL, &pipeline));
    }

    void CreateCommandBuffer() {
      /*
       * TODO  (kyawakyawa) : add japanese comment
       We are getting closer to the end. In order to send commands to the device(GPU),
       we must first record commands into a command buffer.
       To allocate a command buffer, we must first create a command pool. So let us do that.
       */
      VkCommandPoolCreateInfo command_pool_create_info = {};
      command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      command_pool_create_info.flags = 0;
      /*
       * コマンドプールのキューファミリー
       * 全てのコマンドバッファはこのコマンドプールから確保され、
       * このファミリーのキューにのみサブミットされなければならない
       */
      command_pool_create_info.queueFamilyIndex = queue_family_index;
      VK_CHECK_RESULT(vkCreateCommandPool(device, &command_pool_create_info, nullptr, &command_pool));

      /*
       * ここで、コマンドプールからコマンドバッファを確保する
       */
      VkCommandBufferAllocateInfo command_buffer_allocate_info = {};
      command_buffer_allocate_info.sType       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      command_buffer_allocate_info.commandPool = command_pool; // どのコマンドプールからコマンドバッファを確保するか指定する 
      /*
       * もし、コマンドバッファがプライマリーだったら(TODO (kyawakyawa) プライマリーについて書く)、直接キューにサブミットされる
       * セカンダリ のバッファはプライマリーバッファから呼ばれなければならず
       * キューに直接サブミット出来ない
       * 簡単にするためこのプログラムではプライマリーコマンドバッファを使う
       */
      command_buffer_allocate_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      command_buffer_allocate_info.commandBufferCount = 1; // 一つのコマンドバッファを確保する
      VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &command_buffer_allocate_info, &command_buffer)); // コマンドバッファを取得する

      /*
       * 新しく確保したコマンドバッファにコマンドを記録する
       */
      VkCommandBufferBeginInfo begin_info = {};
      begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // このプログラムではバッファは一度だけサブミットされ一度だけ使われる
      VK_CHECK_RESULT(vkBeginCommandBuffer(command_buffer, &begin_info)); // コマンドの記録を開始する

      /*
       * バッファにコマンド送信する(Dispatch)前にパイプラインとdescriptor setを紐付ける必要がある
       *
       * 検証レイヤーはこれを忘れると警告を出さなくなるので忘れないように注意する必要がある
       */
      vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

      /*
       * ckCmdDispatch関数が呼ばれるとcomputeパイプラインが始まり
       * computeシェーダーが実行される
       * ワークグループの数を引数で指定される
       * OpenGLに慣れている人ならこれは新しいことでは無いはずである
       */
      vkCmdDispatch(command_buffer, (uint32_t)ceil(kWidth / float(kWorkgroupeSize)), (uint32_t)ceil(kHeight / float(kWorkgroupeSize)), 1);

      VK_CHECK_RESULT(vkEndCommandBuffer(command_buffer)); // コマンドの記録を終了する
    }

    void RunCommandBuffer() {
      /*
       * コマンドが記録されたコマンドバッファをキューにサブミットする
       */
      VkSubmitInfo submit_info = {};
      submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submit_info.commandBufferCount = 1;               // 一つのコマンドバッファをサブミットする
      submit_info.pCommandBuffers    = &command_buffer; // サブミットするコマンドバッファ

      /*
       * フェンスを作成する
       */
      VkFence fence;
      VkFenceCreateInfo fence_create_info = {};
      fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
      fence_create_info.flags = 0;
      VK_CHECK_RESULT(vkCreateFence(device, &fence_create_info, nullptr, &fence));

      /*
       * キュー上のコマンドバッファをサブミットすると同時に
       * フェンスを与える
       */
      VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submit_info, fence));
      /*
       * フェンスが通知されるまでコマンドは実効を終了しない
       * 従ってここで待つ
       * この直後にGPUからバッファを読み取る
       * フェンスを待たないと、コマンドが実行を終了したかどうかはわからない
       */
      VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000));

      vkDestroyFence(device, fence, NULL);
    }


    void SaveRenderedImage() {
      void* mapped_memory = nullptr;
      // バッファメモリを割り当てて、CPU上で読めるようなる
      vkMapMemory(device, buffer_memory, 0, buffer_size, 0, &mapped_memory);
      Pixel* pmapped_memory = (Pixel *)mapped_memory;

      /*
       * バッファからカラーデータを取得し、1byteに変換する
       * std::vector に保存する
       */
      std::vector<unsigned char> image;
      image.reserve(kWidth * kHeight * 4);
      for (int i = 0; i < kWidth * kHeight; i += 1) {
        image.push_back((unsigned char)(255.0f * (pmapped_memory[i].r)));
        image.push_back((unsigned char)(255.0f * (pmapped_memory[i].g)));
        image.push_back((unsigned char)(255.0f * (pmapped_memory[i].b)));
        image.push_back((unsigned char)(255.0f * (pmapped_memory[i].a)));
      }
      // 読み終わったら解放する
      vkUnmapMemory(device, buffer_memory);

      // 取得したカラーデータをpngにして保存する
      unsigned error = lodepng::encode("mandelbrot.png", image, kWidth, kHeight);
      if (error) printf("encoder error %d: %s", error, lodepng_error_text(error));
    }

    void CleanUp() {
      /*
       * Vulkanリソースを解放する
       */

      if (kEnableValidationLayers) {
        // コールバック関数を破棄する
        auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
        if (func == nullptr) {
          throw std::runtime_error("Could not load vkDestroyDebugReportCallbackEXT");
        }
        func(instance, debug_report_callback, nullptr);
      }

      vkFreeMemory                (device  , buffer_memory        , nullptr);
      vkDestroyBuffer             (device  , buffer               , nullptr);	
      vkDestroyShaderModule       (device  , compute_shader_module, nullptr);
      vkDestroyDescriptorPool     (device  , descriptor_pool      , nullptr);
      vkDestroyDescriptorSetLayout(device  , descriptor_set_layout, nullptr);
      vkDestroyPipelineLayout     (device  , pipeline_layout      , nullptr);
      vkDestroyPipeline           (device  , pipeline             , nullptr);
      vkDestroyCommandPool        (device  , command_pool         , nullptr);	
      vkDestroyDevice             (device  , nullptr);
      vkDestroyInstance           (instance, nullptr);		
    }
    static VKAPI_ATTR VkBool32 VKAPI_CALL DebugReportCallbackFn(
        VkDebugReportFlagsEXT                       flags,
        VkDebugReportObjectTypeEXT                  objectType,
        uint64_t                                    object,
        size_t                                      location,
        int32_t                                     messageCode,
        const char*                                 pLayerPrefix,
        const char*                                 pMessage,
        void*                                       pUserData) {

      printf("Debug Report: %s: %s\n", pLayerPrefix, pMessage);

      return VK_FALSE;
    }

    // 計算処理をサポートしているキューファミリーのインデックス(添字番号)を返す
    uint32_t GetComputeQueueFamilyIndex() {
      uint32_t queue_family_count;

      vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);

      // 全てのキューファミリーを検索する
      std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
      vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());

      // 計算をサポートするキューファミリーを見つける
      uint32_t i = 0;
      for (; i < queue_families.size(); ++i) {
        VkQueueFamilyProperties props = queue_families[i];

        if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
          // 計算用のキューを見つけたら終了
          break;
        }
      }

      if (i == queue_families.size()) {
        throw std::runtime_error("could not find a queue family that supports operations");
      }

      return i;
    }

    // find memory type with desired properties.
    // 目的のプロパティ(特性)を持つメモリタイプを見つける
    uint32_t FindMemoryType(uint32_t memory_type_bits, VkMemoryPropertyFlags properties) {
      VkPhysicalDeviceMemoryProperties memory_properties;

      vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

      /*
       * どのように見つけるかについては
       * ドキュメントのVkPhysicalDeviceMemoryPropertiesにところに詳しい説明がある
       */
      for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        if ((memory_type_bits & (1 << i)) &&
            ((memory_properties.memoryTypes[i].propertyFlags & properties) == properties))
          return i;
      }
      return -1;
    }


    /*
     * char型の配列にファイルを読み uint32_t* に変換して返す
     * データはuint32_tの配列に収まるようにパディング(データを固定長として扱いたいときに、
     * 短いデータの前や後に無意味なデータを追加して長さを合わせる処理のこと)
     * される
     */
    uint32_t* ReadFile(uint32_t& length, const char* filename) {

      FILE* fp = fopen(filename, "rb");
      if (fp == NULL) {
        printf("Could not find or open file: %s\n", filename);
      }

      // ファイルのサイズを取得する
      fseek(fp, 0, SEEK_END);
      long filesize = ftell(fp);
      fseek(fp, 0, SEEK_SET);

      long filesizepadded = long(ceil(filesize / 4.0)) * 4;

      // ファイルの内容を読む
      char *str = new char[filesizepadded];
      fread(str, filesize, sizeof(char), fp);
      fclose(fp);

      // データをパディングする
      for (int i = filesize; i < filesizepadded; i++) {
        str[i] = 0;
      }

      length = filesizepadded;
      return (uint32_t *)str;
    }
};

int main() {
  ComputeApplication app;

  try {
    app.Run();
  }
  catch (const std::runtime_error& e) {
    printf("%s\n", e.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
