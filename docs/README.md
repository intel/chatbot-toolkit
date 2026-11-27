# Intel® Chatbot Toolkit Documentation

The technical documentation for Intel® Chatbot Toolkit. Start with **Meet** for the
what and why, **Get started** to stand it up, **Deploy** to choose a pipeline and go
further, then customize, operate, and look things up as needed.

> [!IMPORTANT]
> [Toolkit Manager](https://github.com/intel/toolkit-manager) installs and operates this toolkit. Clone **that** repository, run every command from its root, and `init erag` will fetch this one into `ext/` for you. See [Meet](overview/meet.md#where-the-pieces-live).

## Meet

| Guide | What it covers |
|-------|----------------|
| [Meet Intel® Chatbot Toolkit](overview/meet.md) | What it is, why it exists, when to choose it, and which repository to start from |
| [FAQ](faq-and-troubleshooting/frequently-asked-questions.md) | Short answers to the questions that come up most |
| [Glossary](reference/glossary.md) | Every term used in these guides, defined once |

## Get started

Check your machine meets the bar, then deploy the stack and ingest a document.

| Guide | What it covers |
|-------|----------------|
| [Prerequisites](getting-started/prerequisites.md) | RAG hardware sizing, software access, and the limited single-user option |
| [Quick Start](getting-started/quickstart.md) | `init erag` to a working assistant, end to end |

## Deploy

Install the toolkit, pick a pipeline, and take it onto a partner platform.

| Guide | What it covers |
|-------|----------------|
| [Deploy Intel® Chatbot Toolkit](deployment/install-rag.md) | Step-by-step install, endpoints, credentials, and teardown |
| [Pipelines](deployment/pipelines.md) | The five flavours, ChatQnA variants, and how to switch between them |
| [Deploy on VMware](deployment/vmware.md) | Running the chatbot on VMware vSphere |

## Customize

Change models, wire up identity and external data sources, and tune every knob.

| Guide | What it covers |
|-------|----------------|
| [Configuration](configuration/configuration.md) | Every option in `config.erag.yaml`, and how overrides resolve |
| [Models](configuration/models.md) | Changing the LLM, embedding, and reranking models; multilingual and accuracy features |
| [Authentication](configuration/auth.md) | Keycloak SSO with Microsoft Entra ID, multi-factor auth, and Active Directory federation |
| [SharePoint](configuration/sharepoint.md) | Ingesting from SharePoint Online, scheduled sync, and per-user site filtering |
| [MCP Integration](configuration/mcp.md) | Exposing retrieval and ingestion to AI agents over Model Context Protocol |
| [Object Store](configuration/object-store.md) | EDP storage backends, and serving documents from NetApp ONTAP S3 |
| [AWS S3 and SQS Setup](configuration/aws-s3.md) | Creating the buckets, queue, and IAM user for external S3 with the AWS CLI |
| [Building Images](configuration/images.md) | Building the component images locally and pointing the deployment at your registry |

## Operate

Watch it, tune it, and fix it once it is running.

| Guide | What it covers |
|-------|----------------|
| [Telemetry](configuration/telemetry.md) | The RAG Grafana dashboards and where to find logs |
| [Performance](configuration/performance.md) | Per-step resources, replica sizing, autoscaling, and vector-database tuning |
| [Backup and Restore](deployment/backup-and-restore.md) | Enabling it, running a backup or restore, what is and is not captured, and the identity realm |
| [Troubleshooting](faq-and-troubleshooting/troubleshooting.md) | Common symptoms, their causes, and the debug tool |

## Reference

How the toolkit is built and what it exposes.

| Guide | What it covers |
|-------|----------------|
| [Architecture](reference/architecture.md) | Components, roles, namespaces, pipeline composition, and request flow |
| [Container Images](reference/container-images.md) | Every image the deployment pulls, with its build context |
| [Service Mesh](reference/service-mesh.md) | How RAG namespaces join the Istio ambient mesh, and the current mTLS posture |
| [Notices](reference/notices.md) | Third-party software, model card guidance, and Intel's Human Rights Principles |

## User guides

Feature and usage walkthroughs for each app, with screenshots.

| Guide | App |
|-------|-----|
| [AudioQnA User Guide](Intel_Chatbot_Toolkit_AudioQnA_User_Guide.pdf) | Voice question answering |
| [ChatQnA User Guide](Intel_Chatbot_Toolkit_ChatQnA_User_Guide.pdf) | Conversational retrieval |
| [DocSum User Guide](Intel_Chatbot_Toolkit_DocSum_User_Guide.pdf) | Document summarization |

## Publications

- [Lenovo Validated Design: AI POD Mini for Enterprise RAG Implementation](https://lenovopress.lenovo.com/lp2417-lenovo-validated-design-ai-pod-mini-for-enterprise-rag-implementation)
- [Accelerate AI Value Creation with Nutanix and Intel® AI for Enterprise RAG](https://www.youtube.com/watch?v=7ghQiKXrzew)
- [Converging Paradigms: Architecting a Hybrid and Open Platform for Unified HPC and AI Workloads](https://www.intel.com/content/www/us/en/content-details/913576/converging-paradigms-architecting-a-hybrid-and-open-platform-for-unified-hpc-and-ai-workloads.html)
- [Starting With the End in Mind: Intel and Nutanix's Blueprint for an Enterprise-Grade RAG Chatbot](https://www.intel.com/content/www/us/en/content-details/898778/starting-with-the-end-in-mind-intel-and-nutanix-s-blueprint-for-an-enterprise-grade-rag-chatbot.html)
- [Securing Enterprise RAG Deployments](https://www.intel.com/content/www/us/en/content-details/870124/securing-enterprise-rag-deployments.html)
- [NetApp AIPod Mini - Deployment Automation](https://community.netapp.com/t5/Tech-ONTAP-Blogs/NetApp-AIPod-Mini-Deployment-Automation/ba-p/463257)
