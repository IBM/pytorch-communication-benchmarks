# pytorch communication benchmarks 
This repository contains simple benchmarks for the most common collective communication calls that are used
in AI training jobs.  These include allreduce, allgather, and reduce-scatter operations.  The benchmarks
are implemented in python scripts that use native PyTorch calls for the collective communication routines.
This makes the tests portable to a variety of systems using any of the backends supported by PyTorch.
It is expected that in most cases users will choose the "nccl" backend, and so that is specified in the
python scripts, but this can be adjusted if desired.  The benchmarks will use the communication library that
is built into your PyTorch distribution, and jobs are launched using your preferred torch.distributed()
launch mechanism.  This ensures that the benchmarks reflect the performance that can be achieved in your
training jobs.

The main benchmark codes are allreduce-loop.py, allgather-loop.py, and reduce-scatter-loop.py.  These codes
loop through a list of global array dimensions and print the average, min, and max bandwidths in units of
GB/sec (10^9 bytes per second) for some number of iterations.  The default number of iterations varies with 
the size of the array, and was chosen to make the tests run quickly on current high-end hardware.  One can
add a command-line option " -m multiplier " to multiply the iteration counts by a constant factor.  The
range of global array dimensions is hard-coded to 8 MB - 10000 MB, which covers the sizes that are commonly
encountered in AI training jobs.  The list of array sizes is close to evenly spaced on a log scale, with
10 data points per decade.  This provides finer granularity than typical power-of-two increments, and is 
suited for plotting the results with a log scale for the X-axis.  These three codes, allreduce-loop.py, 
allgather-loop.py, and reduce-scatter-loop.py have a single communicating group, containing all of the 
torch.distributed() workers.

Job launching is discussed in more detail later, but launches using mpirun, for example, are:

mpirun -np 512 helper.sh python allreduce-loop.py <br />
mpirun -np 512 helper.sh python allgather-loop.py <br />
mpirun -np 512 helper.sh python reduce-scatter-loop.py <br />

where a helper script, helper.sh, is used to set the variables that are needed for the torch.distributed()
environment : MASTER_ADDRESS, MASTER_PORT, RANK, LOCAL_RANK, WORLD_SIZE.  

Some AI training frameworks support multiple dimensions of parallelization, and the communication patterns 
for these frameworks are more complex.  This repository contains a megatron-allreduce.py script that uses
three levels of parallelization, with tensor, pipeline, and data-parallel groups, as in Megatron-LM.  In this
case one of the main collective communication calls is allreduce within each data-parallel group, and there
will be many independent and possibly concurrent calls to allreduce for the different data-parallel groups.
The number of participants in each data-parallel group is the world size divided by the product of the 
tensor-parallel and pipeline-parallel dimensions.  This can be much smaller than the world size, and that can
improve scaling by reducing the effect of latency.  Since there are multiple independent communicating groups,
we add barrier synchronization and take the effective communication time to be the time from when all groups
start until the last group finishes.  It is also of interest to determine whether each group takes about the
same time, or if one or more groups is causing a delay.  To check on this, the megatron-allreduce.py code
has the group leader for each group report separately on the time spent in allreduce calls.  To launch the
job, one needs to specify the dimensions for tensor and pipeline parallelism :

mpirun -np 512 helper.sh python megatron-allreduce.py -t 4 -p 8 <br />

where the -t option is for tensor-parallel and the -p option is for pipeline-parallel.  For the megatron code,
world_size = tp_size * dp_size * pp_size.  Communication for tensor-parallelism is very fine-grained, 
so that is normally limited to within a node.  The default ordering of ranks is "tensor, data, pipeline"  
but the code also supports an alternative "tensor, pipeline, data" ordering, by adding : --order tpd.

Work in progress ...




<!-- Not always needed, but a scope helps the user understand in a short sentence like below, why this repo exists -->
## Scope

The purpose of this project is to provide a template for new open source repositories.

<!-- A more detailed Usage or detailed explanation of the repository here -->
## Usage

This repository contains some example best practices for open source repositories:

* [LICENSE](LICENSE)
* [README.md](README.md)
* [CONTRIBUTING.md](CONTRIBUTING.md)
* [MAINTAINERS.md](MAINTAINERS.md)
<!-- A Changelog allows you to track major changes and things that happen, https://github.com/github-changelog-generator/github-changelog-generator can help automate the process -->
* [CHANGELOG.md](CHANGELOG.md)

> These are optional

<!-- The following are OPTIONAL, but strongly suggested to have in your repository. -->
* [dco.yml](.github/dco.yml) - This enables DCO bot for you, please take a look https://github.com/probot/dco for more details.
* [travis.yml](.travis.yml) - This is a example `.travis.yml`, please take a look https://docs.travis-ci.com/user/tutorial/ for more details.

These may be copied into a new or existing project to make it easier for developers not on a project team to collaborate.

<!-- A notes section is useful for anything that isn't covered in the Usage or Scope. Like what we have below. -->
## Notes

**NOTE: While this boilerplate project uses the Apache 2.0 license, when
establishing a new repo using this template, please use the
license that was approved for your project.**

**NOTE: This repository has been configured with the [DCO bot](https://github.com/probot/dco).
When you set up a new repository that uses the Apache license, you should
use the DCO to manage contributions. The DCO bot will help enforce that.
Please contact one of the IBM GH Org stewards.**

<!-- Questions can be useful but optional, this gives you a place to say, "This is how to contact this project maintainers or create PRs -->
If you have any questions or issues you can create a new [issue here][issues].

Pull requests are very welcome! Make sure your patches are well tested.
Ideally create a topic branch for every separate change you make. For
example:

1. Fork the repo
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Added some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new Pull Request

## License

All source files must include a Copyright and License header. The SPDX license header is 
preferred because it can be easily scanned.

If you would like to see the detailed LICENSE click [here](LICENSE).

```text
#
# Copyright IBM Corp. {Year project was created} - {Current Year}
# SPDX-License-Identifier: Apache-2.0
#
```
## Authors

Optionally, you may include a list of authors, though this is redundant with the built-in
GitHub list of contributors.

- Author: New OpenSource IBMer <new-opensource-ibmer@ibm.com>

[issues]: https://github.com/IBM/repo-template/issues/new
