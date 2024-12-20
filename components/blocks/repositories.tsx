import {
  PageBlocksRepositories,
  PageBlocksRepositoriesRepositories,
} from "../../tina/__generated__/types";
import { tinaField } from "tinacms/dist/react";
import type { TinaTemplate } from "tinacms";
import { ImageLink } from "../util/image-link";
import { basePath } from "../util/url-helper";

const RepoItem = ({
  data,
  language,
}: {
  data: PageBlocksRepositoriesRepositories;
  language: string;
}) => {
  return (
    <div
      data-aos="fade-up"
      data-aos-duration="1000"
      data-tina-field={tinaField(data, "bgImage")}
      className="h-[21rem] bg-cover bg-no-repeat bg-center border-[#EBF1F5] border py-11 px-12 relative"
      style={{
        backgroundImage: `url(${data.bgImage || "default-image-url"})`,
      }}
    >
      {data[`repoName${language}`] && (
        <div
          className="text-base-black font-bold text-3xl leading-10"
          data-tina-field={tinaField(data, "repoNameen")}
        >
          {data[`repoName${language}`]}
        </div>
      )}
      {data[`repoText${language}`] && (
        <div
          className="text-des-blue text-base mt-3 max-w-108 line-clamp-5 h-32"
          data-tina-field={tinaField(data, "repoTexten")}
        >
          {data[`repoText${language}`]}
        </div>
      )}
      {data.linkImage && (
        <ImageLink
          src={data.linkImage.src}
          width={192}
          height={148}
          href={data.linkImage.href}
          className="absolute bottom-11 left-12 w-48 h-12 hover:cursor-pointer"
          tinaField={tinaField(data, "linkImage")}
        />
      )}
    </div>
  );
};

export const Repositories = ({
  data,
  language,
}: {
  data: PageBlocksRepositories;
  language: string;
}) => {
  return (
    <div className="max-w-320 mx-auto px-3">
      <div className="h-px" id="datasets"></div>
      <div className="mt-22">
        {data[`title${language}`] && (
          <div
            className="text-base-black text-5sm font-semibold max-w-191 mx-auto text-center"
            data-tina-field={tinaField(data, "titleen")}
          >
            {data[`title${language}`]}
          </div>
        )}
        <div className="mt-10 mb-22">
          <div className="!grid gap-6 grid-cols-1 lg:grid-cols-3">
            {data.repositories &&
              data.repositories.map((item, index) => {
                return <RepoItem key={index} data={item} language={language} />;
              })}
          </div>
        </div>
      </div>
    </div>
  );
};

const defaultRepository = {
  repoNameen: "taiadv.vision",
  repoNamezh: "taiadv.vision",
  repoTexten:
    "The taiadv.vision toolbox integrates all the methods used to create the adversarial image datasets and benchmarks on this platform.",
  repoTextzh:
    "The taiadv.vision toolbox integrates all the methods used to create the adversarial image datasets and benchmarks on this platform.",
  linkImage: {
    src: "/uploads/GitHubButton.png",
    alt: "taiadv.vision",
    href: "https://github.com/OpenTAI/taiadv",
  },
  bgImage: "/uploads/GitHubBackground1.jpg",
};

export const repositoriesBlockSchema: TinaTemplate = {
  name: "repositories",
  label: "Repositories",
  ui: {
    previewSrc: `${basePath}/blocks/repositories.png`,
    defaultItem: {
      title: "Here's repository title",
    },
  },
  fields: [
    {
      type: "string",
      label: "Title-En",
      name: "titleen",
    },

    {
      type: "string",
      label: "Title-Zh",
      name: "titlezh",
    },
    {
      type: "object",
      label: "Repositories Items",
      name: "repositories",
      list: true,
      ui: {
        itemProps: (item) => {
          return {
            label: item?.repoNameen,
          };
        },
        defaultItem: {
          ...defaultRepository,
        },
      },
      fields: [
        {
          type: "string",
          label: "Repository English Name",
          name: "repoNameen",
        },
        {
          type: "string",
          label: "Repository Chinese Name",
          name: "repoNamezh",
        },
        {
          type: "string",
          label: "Repository English Text",
          name: "repoTexten",
          ui: {
            component: "textarea",
          },
        },
        {
          type: "string",
          label: "Repository Chinese Text",
          name: "repoTextzh",
          ui: {
            component: "textarea",
          },
        },
        {
          type: "object",
          label: "Link Image",
          name: "linkImage",
          fields: [
            {
              name: "src",
              label: "Image Source",
              type: "image",
            },
            {
              name: "alt",
              label: "Alt Text",
              type: "string",
            },
            {
              name: "href",
              label: "Image Link",
              type: "string",
            },
          ],
        },
        {
          type: "image",
          label: "Background Image",
          name: "bgImage",
          ui: {
            component: "image",
          },
        },
      ],
    },
  ],
};
