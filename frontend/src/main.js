import App from './App.svelte';
import 'bootstrap/dist/css/bootstrap.min.css'
import 'bootstrap/dist/js/bootstrap.min.js'

const app = new App({
	target: document.body,
	props: {
		house_list : [
			{
				name: "DRAWER", 
				img_url: "https://image.ohou.se/i/bucketplace-v2-development/uploads/productions/1418699310224_9P6gfFs.jpg?gif=1&w=480&h=480&c=c",
				price: "66,808",
				item: 12345,
			},
			{
				name: "LERBERG 철제선반 L 4colors", 
				img_url: "https://image.ohou.se/i/bucketplace-v2-development/uploads/productions/1471233759715_2916.jpg?gif=1&w=480&h=480&c=c",
				price: "54,000",
				item: 12345,
			},
			{
				name: "[오늘의딜][10%쿠폰/사은품증정] 부드러운 카스테라 옥수수솜 간절기/사계절 차렵이불세트", 
				img_url: "https://image.ohou.se/i/bucketplace-v2-development/uploads/productions/1418719569453_3Z.jpg?gif=1&w=480&h=480&c=c",
				price: "미입점",
				item: 12345,
			},
			{
				name: "모아나 아쿠아텍스 3인용 소파(스툴포함)", 
				img_url: "https://image.ohou.se/i/bucketplace-v2-development/uploads/productions/1418804120564_7e.jpg?gif=1&w=480&h=480&c=c",
				price: "566,000",
				item: 12345,
			},
			{
				name: "[단하루!][단독/NEW한파용] 정말정말 부드러운 두부이불 차렵이불세트", 
				img_url: "https://image.ohou.se/i/bucketplace-v2-development/uploads/productions/1418699889445_Axq.jpg?gif=1&w=480&h=480&c=c",
				price: "12,000",
				item: 12345,
			},
			{
				name: "[13%쿠폰] 1/9단하루! NEW 꼼므 1200 아이책상+의자세트 (화이트/베이지)", 
				img_url: "https://image.ohou.se/i/bucketplace-v2-development/uploads/productions/1418699964946_u.jpg?gif=1&w=480&h=480&c=c",
				price: "54,000",
				item: 12345,
			},
			{
				name: "[오늘의딜][10%쿠폰/사은품증정] 부드러운 카스테라 옥수수솜 간절기/사계절 차렵이불세트", 
				img_url: "https://image.ohou.se/i/bucketplace-v2-development/uploads/productions/1418719569453_3Z.jpg?gif=1&w=480&h=480&c=c",
				price: "24,400",
				item: 12345,
			},
			{
				name: "모아나 아쿠아텍스 3인용 소파(스툴포함)", 
				img_url: "https://image.ohou.se/i/bucketplace-v2-development/uploads/productions/1418804120564_7e.jpg?gif=1&w=480&h=480&c=c",
				price: "미입점",
				item: 12345,
			},
		]
	}
});

export default app;