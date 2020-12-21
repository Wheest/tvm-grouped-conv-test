def get_device_info(device):
    devices = {
        'i7_cpu': {
            'name': 'felicite',
            'host_type': 'local',
            'address': None,
            'port': None,
            'key': None,
            'target': 'llvm',
            'target_string': 'llvm -mtriple=x86_64-linux-gnu -mcpu=core-avx2',

        },
        'hikey_cpu': {
            'name': 'hikey_cpu',
            'address': '130.209.241.132',
            'port': 2083,
            'key': 'btp5a6yUVr0nwLB2kz',
            'target': 'llvm',
            'target_string': 'llvm -mtriple=aarch64-linux-gnu -mattr=+neon',
            'host_type': 'remote',
        },
        'hikey_mali': {
            'name': 'hikey_mali',
            'address': '130.209.241.132',
            'port': 2083,
            'key': 'btp5a6yUVr0nwLB2kz',
            'target': 'opencl -device=mali',
            'target_string': 'llvm -mtriple=aarch64-linux-gnu -mattr=+neon',
            'host_type': 'remote',
        },
    }
    return devices[device]
