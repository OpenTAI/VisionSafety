import Link from "next/link";
import Image from "next/image";

export const ImageLink = ({
  src,
  width,
  height,
  href = "",
  className = "",
  tinaField = "",
}) => {
  if (href) {
    return (
      <Link href={href} target="_blank">
        <Image
          src={src}
          className={className}
          alt=""
          width={width}
          height={height}
          data-tina-field={tinaField}
        />
      </Link>
    );
  } else {
    return (
      <Image
        src={src}
        className={className}
        alt=""
        width={width}
        height={height}
        data-tina-field={tinaField}
      />
    );
  }
};
