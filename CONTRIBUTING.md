# Contributing to examples

We invite contributions and suggessions to this project and want to make engagement as easy as possible. While this
effort focused on foundational work to integrate several open-source frameworks, it is our hope that the community can
leverage what was built for future and increased 5G cybersecurity research.

## Pull Requests

We actively welcome your pull requests but note that initial support will be required by the community to identify 
near term priorities. The first step would be to ideate and collect issues for future build out.

### New Issues

1. Create a GitHub issue proposing a new example and make sure it's substantially different from an existing one. 
Ensure the issue is reasonably scoped and please include a completion criterion; i.e., what will the outcome look like? 
2. Fork the repo and create your branch from `main`. Use a descriptive naming for your feature branch or tie it to an 
issue, e.g., 42_feature_add_UEs.
3. Consider adding tests and simple instructions on how to validate the new code.
4. If the addition is extensive enough, consider creating a `README.md` in the folder.

### Suggested Checklist 
This repository is dependent on several other open source projects so version control and compatibility are important to 
consider. Here are some things that should be included in a contribution for reproducibility: 

- Core network config files
    - Modified configmaps 
    - Modified HPA rules 
    - Modified deployments 
- The UE simulator / emulator used
    - Ex: UERANSIM 
    - Configuration files 
    - Description of the scenario implemented / code to implement the scenario 
    - Data collection pipeline for the given simulator / emulator
- Open5GS version and justification (if changed)
- ONAP version and justification (if changed)


## License Agreement for all Contributors

By contributing to examples, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.

All contributors agree to the license agreement (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).
