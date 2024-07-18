import { Content } from "./blocks/content";
import { Features } from "./blocks/features";
import { Hero } from "./blocks/hero";
import { Testimonial } from "./blocks/testimonial";
import { Repositories } from "./blocks/repositories";
import { Tables } from "./blocks/tables";
import { tinaField } from "tinacms/dist/react";

export const Blocks = (props) => {
  return (
    <>
      {props.blocks
        ? props.blocks.map(function (block, i) {
            return (
              <div key={i} data-tina-field={tinaField(block)}>
                <Block {...block} language={props.language} />
              </div>
            );
          })
        : null}
    </>
  );
};

const Block = (block) => {
  switch (block.__typename) {
    case "PageBlocksContent":
      return <Content data={block} />;
    case "PageBlocksHero":
      return <Hero data={block} language={block.language} />;
    case "PageBlocksFeatures":
      return <Features data={block} language={block.language} />;
    case "PageBlocksTestimonial":
      return <Testimonial data={block} language={block.language} />;
    case "PageBlocksRepositories":
      return <Repositories data={block} language={block.language} />;
    case "PageBlocksTable":
      return <Tables data={block} language={block.language} />;
    default:
      return null;
  }
};
