UPDATE LOG
=

Version 0.1 (Start Date: 09/06/2019)
-

**2019.10.17**
- (TODO) Add network parameters and flops accounting.
- (TODO) Add global args parameters config file.

**2019.09.11**
- (TODO) Modify coding style.
- (TODO) Rewrite metrics module structure.
- (TODO) Rewrite metrics implementation by class to replace
         original function.


**2019.09.06**

- (Finished) In dl library, argument `--gpu` is going to be updated for supporting list argument.
like `--gpu 0 1 2`, which will use GPUs whose idx in `0, 1, 2`.
- (Finished) Updating checkpoints for support apex state dict.