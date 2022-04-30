# @@@@ add resume test here?
# in theory, w/ shuffling at dataloader level, non-persist workers, & worker inits, 100% reprod?
# https://github.com/facebookresearch/hydra/issues/1805
# https://github.com/facebookresearch/hydra/pull/2098  (solution to above)
# https://deploy-preview-2098--hydra-preview.netlify.app/docs/next/experimental/rerun/ (doc for above)
# old:
# https://github.com/facebookresearch/hydra/issues/1576
# https://stackoverflow.com/questions/67170653/how-to-load-hydra-parameters-from-previous-jobs-without-having-to-use-argparse/67172466?noredirect=1

# ASSERT THAT RESUMING IS ONLY SUPPORTED FOR SINGLE RUN (NOT MULTI RUN); see #1805 and #2098 above
