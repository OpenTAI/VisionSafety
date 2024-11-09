import type { Collection } from "tinacms";
import { heroBlockSchema } from "../../components/blocks/hero";
import { contentBlockSchema } from "../../components/blocks/content";
import { testimonialBlockSchema } from "../../components/blocks/testimonial";
import { featureBlockSchema } from "../../components/blocks/features";
import { repositoriesBlockSchema } from "../../components/blocks/repositories";
import { tablesBlockSchema } from "../../components/blocks/tables";

const Page: Collection = {
  label: "Pages",
  name: "page",
  path: "content/pages",
  ui: {
    router: ({ document }) => {
      if (document._sys.filename === "home") {
        return ``;
      }
      if (document._sys.filename === "leaderboards") {
        return `${process.env.NEXT_PUBLIC_BASE_PATH || ''}/leaderboards`;
      }
      return undefined;
    },
  },
  fields: [
    {
      type: "string",
      label: "Title",
      name: "title",
      description:
        "The title of the page. This is used to display the title in the CMS",
      isTitle: true,
      required: true,
    },
    {
      type: "object",
      list: true,
      name: "blocks",
      label: "Sections",
      ui: {
        visualSelector: true,
      },
      templates: [
        heroBlockSchema,
        // eslint-disable-next-line
        // @ts-ignore
        featureBlockSchema,
        repositoriesBlockSchema,
        contentBlockSchema,
        testimonialBlockSchema,
        tablesBlockSchema,
      ],
    },
  ],
};

export default Page;
