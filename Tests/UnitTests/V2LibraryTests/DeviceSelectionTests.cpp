//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Common.h"

using namespace CNTK;

namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(DeviceSelectionSuite)

BOOST_AUTO_TEST_CASE(TestDefaultDeviceSelection)
{
    auto placeholderDefaultDevice = DeviceDescriptor::DefaultDevice();
    BOOST_TEST((placeholderDefaultDevice.Type() == DeviceKind::AUTO));// default device is a placeholder

    const auto& allDevices = DeviceDescriptor::AllDevices();
    BOOST_TEST((find(allDevices.begin(), allDevices.end(), placeholderDefaultDevice) == allDevices.end()));

    DeviceDescriptor::SetDefaultDevice(placeholderDefaultDevice); // nothing happens here
    BOOST_TEST((DeviceDescriptor::DefaultDevice() == placeholderDefaultDevice)); // DefaultDevice() still returns a placeholder

    auto actualDefaultDevice = DeviceDescriptor::UseDefaultDevice(); // At this point, a physical device is selected
    BOOST_TEST((DeviceDescriptor::DefaultDevice() == actualDefaultDevice));
    BOOST_TEST((actualDefaultDevice != placeholderDefaultDevice));
    BOOST_TEST((find(allDevices.begin(), allDevices.end(), actualDefaultDevice) != allDevices.end()));
}

BOOST_AUTO_TEST_CASE(SetCpuDeviceAsDefault)
{
    auto cpuDevice = DeviceDescriptor::CPUDevice();

    DeviceDescriptor::SetDefaultDevice(cpuDevice);
    BOOST_TEST((DeviceDescriptor::DefaultDevice() == cpuDevice));
    BOOST_TEST((DeviceDescriptor::UseDefaultDevice() == cpuDevice));

    const auto& allDevices = DeviceDescriptor::AllDevices();

#ifdef CPUONLY
    BOOST_TEST((allDevices.size() == 1));
#endif

    if (allDevices.size() > 1)
    {
        auto nonCpuDevice = std::find_if(allDevices.begin(), allDevices.end(),
            [&cpuDevice](const DeviceDescriptor& x)->bool{return x != cpuDevice; });
        
        VerifyException([&nonCpuDevice]() {
            DeviceDescriptor::SetDefaultDevice(*nonCpuDevice);
        }, "Was able to invoke SetDefaultDevice() after UseDefaultDevice().");
    }
}

BOOST_AUTO_TEST_CASE(SetNonCPUDeviceAsDefault)
{
    auto cpuDevice = DeviceDescriptor::CPUDevice();
    const auto& allDevices = DeviceDescriptor::AllDevices();

#ifdef CPUONLY
    BOOST_TEST((allDevices.size() == 1));
#endif

    if (allDevices.size() > 1)
    {
        auto nonCpuDevice = std::find_if(allDevices.begin(), allDevices.end(),
            [&cpuDevice](const DeviceDescriptor& x)->bool {return x != cpuDevice; });

        DeviceDescriptor::SetDefaultDevice(*nonCpuDevice);

        BOOST_TEST((DeviceDescriptor::DefaultDevice() == *nonCpuDevice));
        BOOST_TEST((DeviceDescriptor::UseDefaultDevice() == *nonCpuDevice));

        VerifyException([&cpuDevice]() {
            DeviceDescriptor::SetDefaultDevice(cpuDevice);
        }, "Was able to invoke SetDefaultDevice() after UseDefaultDevice().");
    }
}

BOOST_AUTO_TEST_CASE(TestAllDevicesContainsGPUsAndCPU)
{
    auto cpuDevice = DeviceDescriptor::CPUDevice();
    
    const auto& allDevices = DeviceDescriptor::AllDevices();
    BOOST_TEST((find(allDevices.begin(), allDevices.end(), cpuDevice) != allDevices.end()));

#ifdef CPUONLY
    BOOST_TEST((allDevices.size() == 1));
#endif
    auto numGpuDevices = allDevices.size() - 1;

    VerifyException([&numGpuDevices]() {
        DeviceDescriptor::GPUDevice((unsigned int)numGpuDevices);
    }, "Was able to create GPU device descriptor with invalid id.");

    BOOST_TEST((allDevices.back() == cpuDevice));
}

BOOST_AUTO_TEST_SUITE_END()

}}
