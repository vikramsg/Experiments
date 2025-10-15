import { useRouter } from 'next/router'

export default function BlogPost(): JSX.Element {
  const router = useRouter()
  const { slug } = router.query

  return (
    <div>
      <h1>Blog Post: {slug}</h1>
      <p>You are viewing the post with slug: {slug}</p>
    </div>
  )
}
