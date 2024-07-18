import * as React from "react";
import { Section } from "../util/section";
import type { TinaTemplate } from "tinacms";
import { PageBlocksHero } from "../../tina/__generated__/types";
import { tinaField } from "tinacms/dist/react";
import Image from "next/image";
import headImg from "../../assets/img/headImg.png";

export const Hero = ({
  data,
  language,
}: {
  data: PageBlocksHero;
  language: string;
}) => {
  return (
    <Section>
      <div className="max-w-360 mx-auto relative">
        <Image
          src={headImg}
          className="h-208 absolute -z-10 object-cover object-top"
          alt=""
        />
        <div className="flex items-center flex-col">
          {data[`headline1${language}`] && (
            <div
              className="mt-16 text-base-blue text-6xl font-extralight max-w-168 text-center leading-18 mx-3"
              data-tina-field={tinaField(data, "headline1en")}
            >
              {data[`headline1${language}`]}
            </div>
          )}
          {data[`headline2${language}`] && (
            <div
              className="mt-5 text-3xl font-light leading-9 text-center mx-3"
              data-tina-field={tinaField(data, "headline2en")}
            >
              {data[`headline2${language}`]}
            </div>
          )}
          {data[`buttonText${language}`] && (
            <div
              className="bg-base-blue h-19 w-80 text-white text-2xl font-medium flex items-center justify-center mt-8"
              data-tina-field={tinaField(data, "buttonTexten")}
            >
              {data[`buttonText${language}`]}
            </div>
          )}
        </div>
        <div className="bg-worldImg bg-center h-233 bg-cover mx-auto mt-16 relative">
          <div
            className="text-base-blue text-[42px] font-bold text-center pt-10"
            data-tina-field={tinaField(data, "subtitle1en")}
          >
            {data[`subtitle1${language}`]}
          </div>
          <div
            className="max-w-168 mx-auto text-center mt-2 text-lg text-base-grey"
            data-tina-field={tinaField(data, "subtitle2en")}
          >
            {data[`subtitle2${language}`]}
          </div>
        </div>
        <div className="max-w-320 mx-auto px-3">
          <div className="mt-14">
            <div
              className="text-base-blue text-5sm font-semibold max-w-191 text-center mx-auto leading-14"
              data-tina-field={tinaField(data, "text1en")}
            >
              {data[`text1${language}`]}
            </div>
            <div
              className="text-light-blue text-base max-w-191 text-center mx-auto mt-4"
              data-tina-field={tinaField(data, "text2en")}
            >
              {data[`text2${language}`]}
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
};

export const heroBlockSchema: TinaTemplate = {
  name: "hero",
  label: "Hero",
  ui: {
    previewSrc: "/blocks/hero.png",
    defaultItem: {},
  },
  fields: [
    {
      type: "string",
      label: "Headline-1-En",
      name: "headline1en",
    },
    {
      type: "string",
      label: "Headline-1-Zh",
      name: "headline1zh",
    },
    {
      type: "string",
      label: "Headline-2-En",
      name: "headline2en",
    },
    {
      type: "string",
      label: "Headline-2-Zh",
      name: "headline2zh",
    },
    {
      type: "string",
      label: "Button Text-En",
      name: "buttonTexten",
    },
    {
      type: "string",
      label: "Button Text-Zh",
      name: "buttonTextzh",
    },
    {
      type: "string",
      label: "Subtitle-1-En",
      name: "subtitle1en",
    },
    {
      type: "string",
      label: "Subtitle-1-Zh",
      name: "subtitle1zh",
    },
    {
      type: "string",
      label: "Subtitle-2-En",
      name: "subtitle2en",
    },
    {
      type: "string",
      label: "Subtitle-2-Zh",
      name: "subtitle2zh",
    },
    {
      type: "string",
      label: "Text-1-En",
      name: "text1en",
    },
    {
      type: "string",
      label: "Text-1-Zh",
      name: "text1zh",
    },
    {
      type: "string",
      label: "Text-2-En",
      name: "text2en",
    },
    {
      type: "string",
      label: "Text-2-Zh",
      name: "text2zh",
    },
    {
      type: "object",
      label: "Image",
      name: "image",
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
      ],
    },
  ],
};
