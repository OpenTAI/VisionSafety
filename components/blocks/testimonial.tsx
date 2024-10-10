import React from "react";
import type { TinaTemplate } from "tinacms";
import {
  PageBlocksTestimonial,
  PageBlocksTestimonialItems1,
  PageBlocksTestimonialItems2,
} from "../../tina/__generated__/types";
import { tinaField } from "tinacms/dist/react";
import sparkles from "../../assets/img/sparkles.png";
import stars from "../../assets/img/stars.png";
import Image from "next/image";

export const LeftListItem = ({
  data,
  index,
  language,
}: {
  data: PageBlocksTestimonialItems1;
  index: number;
  language: string;
}) => {
  return (
    <div
      data-aos="fade-up"
      data-aos-duration="1000"
      data-aos-delay={index * 2 * 100}
      key={index}
      className="mt-4 bg-leaderboardsBg1 min-h-40 bg-cover bg-no-repeat bg-center border-[#EBF1F5] border py-5 px-7 relative"
    >
      {data[`title${language}`] && (
        <div
          className="text-base-black font-bold text-2xl leading-7"
          data-tina-field={tinaField(data, "titleen")}
        >
          {data[`title${language}`]}
        </div>
      )}
      {data[`subtitle${language}`] && (
        <div
          className="text-base-black text-lg mt-1 max-w-108 flex items-center"
          data-tina-field={tinaField(data, "subtitleen")}
        >
          <Image src={sparkles} className="w-[18px] h-[16px] mr-1" alt="" />
          {data[`subtitle${language}`]}
        </div>
      )}
      <div className="flex mt-3 justify-between">
        <div>
          {data.modelSum && (
            <span
              className="text-base-blue text-5xl font-bold mr-1"
              data-tina-field={tinaField(data, "modelSum")}
            >
              {data.modelSum}
            </span>
          )}
          <span className="text-light-blue text-lg">models included</span>
        </div>
        <div className="flex flex-col items-end">
          <div className="flex text-base-black">
            <Image src={stars} className="w-[120px] h-[24px] mr-2" alt="" />
            {data.score && (
              <span
                className="text-lg leading-6"
                data-tina-field={tinaField(data, "score")}
              >
                {data.score}
              </span>
            )}
          </div>
          {data[`detail${language}`] && (
            <div data-tina-field={tinaField(data, "detailen")}>
              {data[`detail${language}`]}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export const RightListItem = ({
  data,
  index,
  language,
}: {
  data: PageBlocksTestimonialItems2;
  index: number;
  language: string;
}) => {
  return (
    <div
      data-aos="fade-up"
      data-aos-duration="1000"
      data-aos-delay={index * 2 * 100}
      key={index}
      className="mt-4 bg-leaderboardsBg2 min-h-40 bg-cover bg-no-repeat bg-center border-[#EBF1F5] border py-5 px-7 relative"
    >
      {data[`title${language}`] && (
        <div
          className="text-base-black font-bold text-2xl leading-7"
          data-tina-field={tinaField(data, "titleen")}
        >
          {data[`title${language}`]}
        </div>
      )}
      {data[`subtitle${language}`] && (
        <div
          className="text-base-black text-lg mt-1 max-w-108 flex items-center"
          data-tina-field={tinaField(data, "subtitleen")}
        >
          <Image src={sparkles} className="w-[18px] h-[16px] mr-1" alt="" />
          {data[`subtitle${language}`]}
        </div>
      )}
      <div className="flex mt-3 justify-between">
        <div>
          {data.modelSum && (
            <span
              className="text-base-blue text-5xl font-bold mr-1"
              data-tina-field={tinaField(data, "modelSum")}
            >
              {data.modelSum}
            </span>
          )}
          <span className="text-light-blue text-lg">models included</span>
        </div>
        <div className="flex flex-col items-end">
          <div className="flex text-base-black">
            <Image src={stars} className="w-[120px] h-[24px] mr-2" alt="" />
            {data.score && (
              <span
                className="text-lg leading-6"
                data-tina-field={tinaField(data, "score")}
              >
                {data.score}
              </span>
            )}
          </div>
          {data[`detail${language}`] && (
            <div data-tina-field={tinaField(data, "detailen")}>
              {data[`detail${language}`]}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export const Testimonial = ({
  data,
  language,
}: {
  data: PageBlocksTestimonial;
  language: string;
}) => {
  return (
    <div className="max-w-320 mx-auto px-3">
      <div className="my-24">
        {data[`title${language}`] && (
          <div
            className="text-base-black font-semibold text-4xl text-center"
            data-tina-field={tinaField(data, "titleen")}
          >
            {data[`title${language}`]}
          </div>
        )}
        <div className="mt-10">
          <div className="!grid gap-6 grid-cols-1 lg:grid-cols-2">
            <div>
              {data[`leftListTitle${language}`] && (
                <div
                  className="text-base-blue text-sm w-33 h-8 bg-[#edf1fe] flex items-center justify-center"
                  data-tina-field={tinaField(data, "leftListTitleen")}
                >
                  {data[`leftListTitle${language}`]}
                </div>
              )}
              {data.items1 &&
                data.items1.map((item, index) => {
                  return (
                    <LeftListItem
                      data={item}
                      index={index}
                      key={2 * index}
                      language={language}
                    />
                  );
                })}
            </div>
            <div>
              {data[`rightListTitle${language}`] && (
                <div
                  className="text-base-blue text-sm w-33 h-8 bg-[#edf1fe] flex items-center justify-center"
                  data-tina-field={tinaField(data, "rightListTitleen")}
                >
                  {data[`rightListTitle${language}`]}
                </div>
              )}
              {data.items2 &&
                data.items2.map((item, index) => {
                  return (
                    <RightListItem
                      data={item}
                      index={index}
                      key={2 * index + 1}
                      language={language}
                    />
                  );
                })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const defaultModel = {
  title: "Here's Another Model",
  subtitle: "Large language model",
};

export const testimonialBlockSchema: TinaTemplate = {
  name: "testimonial",
  label: "Testimonial",
  ui: {
    previewSrc: "/VisionSafety/blocks/testimonial.png",
    defaultItem: {
      quote:
        "There are only two hard things in Computer Science: cache invalidation and naming things.",
      author: "Phil Karlton",
      color: "primary",
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
      type: "string",
      label: "Left List Title-En",
      name: "leftListTitleen",
    },
    {
      type: "string",
      label: "Left List Title-Zh",
      name: "leftListTitlezh",
    },
    {
      type: "object",
      label: "Left List Items",
      name: "items1",
      list: true,
      ui: {
        itemProps: (item) => {
          return {
            label: item?.titleen,
          };
        },
        defaultItem: {
          ...defaultModel,
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
          type: "string",
          label: "Subtitle-En",
          name: "subtitleen",
        },
        {
          type: "string",
          label: "Subtitle-Zh",
          name: "subtitlezh",
        },
        {
          type: "number",
          label: "Model Sum",
          name: "modelSum",
        },
        {
          type: "string",
          label: "score",
          name: "score",
        },
        {
          type: "string",
          label: "Detail-En",
          name: "detailen",
        },
        {
          type: "string",
          label: "Detail-Zh",
          name: "detailzh",
        },
      ],
    },
    {
      type: "string",
      label: "Right List Title-En",
      name: "rightListTitleen",
    },
    {
      type: "string",
      label: "Right List Title-Zh",
      name: "rightListTitlezh",
    },
    {
      type: "object",
      label: "Right List Items",
      name: "items2",
      list: true,
      ui: {
        itemProps: (item) => {
          return {
            label: item?.titleen,
          };
        },
        defaultItem: {
          ...defaultModel,
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
          type: "string",
          label: "Subtitle-En",
          name: "subtitleen",
        },
        {
          type: "string",
          label: "Subtitle-Zh",
          name: "subtitlezh",
        },
        {
          type: "number",
          label: "Model Sum",
          name: "modelSum",
        },
        {
          type: "string",
          label: "score",
          name: "score",
        },
        {
          type: "string",
          label: "Detail-En",
          name: "detailen",
        },
        {
          type: "string",
          label: "Detail-Zh",
          name: "detailzh",
        },
      ],
    },
  ],
};
