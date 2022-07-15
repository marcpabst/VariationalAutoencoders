using ProgressMeter
using FileIO
using PyCall
np = pyimport("numpy");

files = ["/path/to/train_data_batch_1.npz",
         "/path/to/train_data_batch_2.npz",
          "/path/to/train_data_batch_3.npz",
         "/path/to/train_data_batch_4.npz",
         "/path/to/train_data_batch_5.npz",
         "/path/to/train_data_batch_6.npz",
         "/path/to/train_data_batch_7.npz",
         "/path/to/train_data_batch_8.npz",
         "/path/to/train_data_batch_9.npz",
         "/path/to/train_data_batch_10.npz"
        ]

function load_data(files)
    start_idx = 1

    # allocate memory
    data = zeros(UInt8, 64, 64, 3, 128123 * length(files))
    labels = zeros(Int, 128123 * length(files))

    @showprogress for (i,file) in enumerate(files)
        _batch = np.load(file, allow_pickle=true)

        _data = get(_batch, :data)
        _labels = get(_batch, :labels)
        
        n_samples = size(_data)[1]
        
        data[:,:,:, start_idx:start_idx+n_samples-1] = permutedims(reshape(_data, :, 64, 64, 3), (3,2,4,1))
        labels[start_idx:start_idx+n_samples-1] = _labels
        start_idx = start_idx + n_samples
        GC.gc()
    end

    return data, labels
end

@info "Starting to load files..."
data, labels = load_data(files)

@info "Saving file..."
FileIO.save("/path/to/train_data.jld2", Dict("data" => data, "labels" => labels); compress=false)