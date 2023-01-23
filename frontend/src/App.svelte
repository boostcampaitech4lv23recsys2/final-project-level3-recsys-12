<script>

	import Router from 'svelte-spa-router'
	import Home from './Home.svelte'
	import Detail from './Detail.svelte'
	import Login from './Login.svelte'
	import SignUp from './SignUp.svelte'
	import Header from './Header.svelte'
	import Footer from './Footer.svelte'
	// import Mypage from './Mypage.svelte'

	const routes = {
		'/': Home,
		'/detail/:id': Detail,
		'/login-user': Login,
		'/signup-user': SignUp,
		// '/mypage': Mypage,
	}
</script>

<Header />
<Router {routes} />
<Footer />